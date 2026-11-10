"""Framework-free reporting/launch controls for the restored v15 protocol."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eegproc.deep_learning.joint_architectures.SICModelv16 import v15_protocol as protocol
from eegproc.deep_learning.joint_architectures.SICModelv16 import run_experiments as cli
ROOT=Path(__file__).resolve().parents[2]
SIC=ROOT/'src/eegproc/deep_learning/joint_architectures/SICModelv16'


class ProtocolControlTests(unittest.TestCase):
    def spec(self):
        return dict(protocol=protocol.PROTOCOL,configuration={'name':'test','hyperparameters':{}},
                    source_epochs=2,target_subjects=[0,1],seed=42,calibration_levels=[[3,6]])

    def save_rows(self,root,omit_last=False):
        for subject,epoch,score in [(0,1,.9),(1,1,.1),(0,2,.6),(1,2,.7)]:
            if omit_last and (subject,epoch)==(1,2):
                continue
            folder=root/f'subject_{subject:02d}'
            folder.mkdir(exist_ok=True)
            model=folder/f'epoch_{epoch:03d}.keras'
            model.write_bytes(b'not loaded in reporting test')
            protocol.write_json(folder/f'epoch_{epoch:03d}.json',dict(target_subject=subject,epoch=epoch,
                decision_threshold=.5,balanced_accuracy=score,checkpoint=str(model.relative_to(root)),
                test_metrics={'balanced_accuracy':score,'accuracy':score},
                y_true=[0,1],probabilities=[[.7,.3],[.2,.8]]))

    def test_selection_uses_one_common_epoch_and_requires_complete_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            protocol.write_json(root/'task.json',self.spec())
            self.save_rows(root,omit_last=True)
            self.assertEqual(protocol.summarize(root)['oracle']['status'],'incomplete')
            self.assertNotIn('final',protocol.publish_final(root))
            self.save_rows(root)
            summary=protocol.summarize(root)
            self.assertEqual(summary['oracle']['best_epoch'],2)
            self.assertAlmostEqual(summary['oracle']['selection_score'],.65)
            self.assertEqual([r['epoch'] for r in summary['oracle']['folds']],[2,2])
            self.assertNotIn('final',protocol.publish_final(root))
            for subject in [0,1]:
                protocol.write_json(root/'calibration'/f'subject_{subject:02d}'/'result.json',dict(
                    protocol=protocol.PROTOCOL,selected_epoch=2,overall={'calibration_levels':{
                        '3':{'calibrated_mean_scores':{'balanced_accuracy':.8,'accuracy':.85}}}}))
            final=protocol.publish_final(root)['final']
            self.assertAlmostEqual(final['mean_balanced_accuracy'],.65)
            self.assertAlmostEqual(final['calibration_levels']['3']['calibrated_mean_scores']['balanced_accuracy'],.8)
            spec=importlib.util.spec_from_file_location('restored_tracker_bridge',ROOT/'experiments/sic_v16/train.py')
            bridge=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bridge)
            result=bridge.tracker_result(protocol.publish_final(root))
            self.assertTrue(result['oracle'])
            self.assertAlmostEqual(result['metrics']['trial_balanced_accuracy']['mean'],.65)

    def test_cli_defaults_dispatch_to_restored_protocol(self):
        with patch.object(protocol,'run',return_value=0) as run:
            code=cli.main(['run','--config-index','0','--output-dir','unused',
                          '--eeg','unused.npy','--labels','unused.npy'])
        self.assertEqual(code,0)
        args=run.call_args.args[0]
        self.assertEqual((args.source_epochs,args.batch_size,args.validation_subjects),(12,64,0))
        self.assertEqual((args.n_jobs,args.gpus_per_fold,args.calibration_epochs),(2,2,1))

    def test_new_protocol_refuses_legacy_data_split_before_writing(self):
        args=argparse.Namespace(configs=SIC/'configurations.json',config_index=0,
            validation_subjects=2,data_subjects=None,output_dir=Path('must-not-exist'))
        with self.assertRaisesRegex(ValueError,'all 22 source subjects'):
            protocol.run(args)
        args.validation_subjects=0
        args.data_subjects=[0,1,2,3]
        with self.assertRaisesRegex(ValueError,'all 22 source subjects'):
            protocol.run(args)

    def test_no_device_allocation_needed_to_publish_completed_runs(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(protocol.launch_stage(Path('unused'),{},'worker',[]),([],False))


if __name__=='__main__':
    unittest.main(verbosity=2)
