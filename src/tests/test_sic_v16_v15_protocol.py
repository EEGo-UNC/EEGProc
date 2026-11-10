"""Restored objective, two-device parity, source selection and v15 calibration.

Run in a fresh Python process. Logical CPUs test sharding without claiming CUDA
validation; SIC16_TEST_GPUS=1 selects the two-GPU cluster preflight instead.
"""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[2]
SIC = ROOT / 'src/eegproc/deep_learning/joint_architectures/SICModelv16'


class RestoredProtocolTests(unittest.TestCase):
    __test__ = False  # device initialization needs a fresh process under pytest
    @classmethod
    def setUpClass(cls):
        if os.environ.get('SIC16_TEST_GPUS') == '1':
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) < 2:
                raise RuntimeError('Restored v16 preflight requires two real GPUs.')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            cls.devices = tuple(d.name for d in tf.config.list_logical_devices('GPU')[:2])
        else:
            tf.config.set_visible_devices([], 'GPU')
            tf.config.set_logical_device_configuration(tf.config.list_physical_devices('CPU')[0],
                [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()])
            cls.devices = tuple(d.name for d in tf.config.list_logical_devices('CPU'))
        from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model
        from eegproc.deep_learning.joint_architectures.SICModelv16 import v15_protocol
        cls.builder = staticmethod(build_sic_model)
        cls.protocol = v15_protocol

    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def tiny(self, **overrides):
        config = dict(conv_filters=[2,4], emb_dim=4, attention_heads=2, feed_forward_dim=8,
            dropout=0., decoder_dropout=0., classifier_rnn_units=[4,3], n_classifier_rnn_layers=2,
            classifier_rnn_dropout=0., n_subject_classes=6, training_method='mldg',
            mldg_meta_train_subjects=4, mldg_meta_test_subjects=2,
            mldg_trials_per_subject=2, mldg_steps_per_epoch=1, vc_beta=.6, vc_lambda=.05,
            vc_alpha=2., vc_logit_scale=16., subject_adversarial_weight=.6,
            subject_loss_weight=.2, reconstruction_loss_weight=.6)
        config.update(overrides)
        return self.builder((1,8,42), **config)

    def test_two_device_objective_and_mldg_updates(self):
        x=tf.constant(np.random.default_rng(2).normal(size=(12,1,8,42)),tf.float32)
        y=tf.constant([0]*4+[1]*4+[0]*2+[1]*2)
        subjects=tf.repeat(tf.range(6),2)
        for transformer, decoder in ((False, False), (False, True), (True, False), (True, True)):
            single,parallel=(self.tiny(use_transformer=transformer, use_decoder=decoder),
                             self.tiny(use_transformer=transformer, use_decoder=decoder))
            self.assertTrue(single.use_recurrent_classifier and single.use_variational_head)
            self.assertTrue(single.subject_adversarial_enabled and single.use_mldg)
            parallel.set_weights(single.get_weights())
            parallel.configure_fold_devices(self.devices,deterministic_execution=True)
            def objective(model):
                with tf.GradientTape() as tape:
                    out=model._encode_mldg(x[:7],training=True)
                    vc=model._vc_components(out['classification_embedding'],out['logits'],y[:7],None,calibration=False)
                    rec=model._reconstruction_components(out,True)
                    sub=model._subject_components(out['pooled_features'],subjects[:7],training=True,use_grl=True)
                    loss=vc['total_loss']+.6*rec['reconstruction_loss']+.2*sub['subject_loss']
                return loss,tape.gradient(loss,model.trainable_variables)
            a,ag=objective(single)
            b,bg=objective(parallel)
            np.testing.assert_allclose(a,b,atol=3e-5,rtol=3e-4)
            for left,right in zip(ag,bg,strict=True):
                self.assertEqual(left is None,right is None)
                if left is not None:
                    np.testing.assert_allclose(tf.convert_to_tensor(left),tf.convert_to_tensor(right),atol=3e-5,rtol=3e-4)
            inputs={'eeg':x,'subject_id':subjects,'mldg_role':tf.constant([0]*8+[1]*4)}
            single._mldg_train_step(inputs,y,None)
            parallel._mldg_train_step(inputs,y,None)
            self.assertEqual(int(parallel.main_optimizer.iterations.numpy()),1)
            for left,right in zip(single.weights,parallel.weights,strict=True):
                np.testing.assert_allclose(left.numpy(),right.numpy(),atol=3e-5,rtol=3e-4)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'restored.keras'
                parallel.save(path)
                loaded=tf.keras.models.load_model(path,compile=True)
                np.testing.assert_allclose(single(x,training=False),loaded(x,training=False),atol=3e-5,rtol=3e-4)
                for a,b in zip(parallel.main_optimizer.variables,loaded.main_optimizer.variables,strict=True):
                    np.testing.assert_array_equal(a.numpy(),b.numpy())

    def test_source_to_common_epoch_and_calibration(self):
        config=json.loads((SIC/'configurations.json').read_text())['base']
        config.update(conv_filters=[2,4],emb_dim=4,attention_heads=2,feed_forward_dim=8,
            dropout=0.,decoder_dropout=0.,classifier_rnn_units=[4,3],classifier_rnn_dropout=0.,
            mldg_meta_train_subjects=2,mldg_meta_test_subjects=1,mldg_trials_per_subject=2,mldg_steps_per_epoch=1)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            (output/'data').mkdir()
            arrays=dict(features=np.random.default_rng(3).normal(size=(16,1,8,42)).astype('float32'),
                labels=np.tile([0,1,0,1],4),subjects=np.repeat(np.arange(4),4),trials=np.tile(np.arange(4),4))
            for name,value in arrays.items():
                np.save(output/'data'/f'{name}.npy',value)
            spec=dict(protocol=self.protocol.PROTOCOL,configuration={'name':'tiny','hyperparameters':config},
                target_subjects=[0],source_epochs=1,batch_size=64,seed=42,deadline=time.time()+600,
                gpus_per_fold=1,calibration_epochs=1,calibration_batch_size=64,calibration_levels=[[2,2]])
            self.protocol.write_json(output/'task.json',spec)
            # Actual gradients, checkpoint IO, selection and v15 calibration;
            # only discovery of physical GPUs is substituted for local testing.
            with patch.object(self.protocol,'configure_devices',return_value=self.devices[:1]):
                self.assertEqual(self.protocol.worker(output/'task.json',0),0)
                self.assertEqual(self.protocol.summarize(output)['oracle']['best_epoch'],1)
                self.assertEqual(self.protocol.calibration_worker(output/'task.json',0),0)
            final=self.protocol.publish_final(output)['final']
            self.assertEqual(final['status'],'completed')
            self.assertIn('2',final['calibration_levels'])
            self.assertEqual(json.loads((output/'subject_00/split.json').read_text())['validation_subjects'],[])

    def test_campaigns_and_baseline_parity(self):
        reference=json.loads((SIC.parent/'SICModelv15/sic_v15_valence_configs.json').read_text())['base']
        manifest=json.loads((SIC/'configurations.json').read_text())
        excluded={'gcn_units','gcn_dropout','gcn_activation','gcn_use_batch_norm','spectral_gru_units',
            'spectral_gru_dropout','mi_n_neighbors','mi_random_state','mi_zero_diagonal','mi_band_reduction',
            'mi_max_observations','bilstm_units','n_bilstm_layers','bilstm_dropout','use_gcn_gru_branch',
            'use_bilstm_branch','joint_reconstruction_auxiliary_weight','joint_reconstruction_initial_alpha'}
        for key,value in reference.items():
            if key not in excluded:
                self.assertEqual(manifest['base'][key],value,key)
        from training_tracker.core import load_campaign,plan_run
        for row in manifest['configurations']:
            campaign=load_campaign(ROOT/'experiments/sic_v16/v15_protocol'/row['name']/'campaign.json')
            self.assertEqual(campaign['defaults']['validation_subjects'],0)
            self.assertEqual(campaign['defaults']['target_subjects'],list(range(23)))
            self.assertNotIn('data_subjects',campaign['defaults'])
            with tempfile.TemporaryDirectory() as folder:
                run=plan_run(campaign,Path(folder)/'runs','valence')
                planned=json.loads((run/'config.json').read_text())
                self.assertTrue(planned['use_recurrent_classifier'])
                self.assertEqual(planned['training_method'],'mldg')
                self.assertIn('#SBATCH --gres=gpu:4',(run/'submit.slurm').read_text())


def test_restored_protocol_in_fresh_process():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__=='__main__':
    unittest.main(verbosity=2)
