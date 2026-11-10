"""Offline displacement audit: integrity, metric spaces, pairing and missingness."""

import json
import shutil

import numpy as np
import pytest

from eegproc.model_explainability.typicality.artifacts import file_sha256, write_json
from eegproc.model_explainability.typicality.coverage import eligibility_records
from eegproc.model_explainability.typicality.displacement_audit import build_audit, collect_audit, DISTANCE_DEFINITION
from eegproc.model_explainability.typicality.results import population_summary


def archive(tmp_path):
    root=tmp_path/'study'
    write_json(root/'study.json',{
        'task':'valence','direction':'0_to_1','source_class':0,'target_class':1,
        'arguments':{'typicality_quantile':.95,'target_probability':.5},
        'folds':[{'subject_id':0,'sha256':'frozen-checkpoint'}],
        'objectives':['base','typicality'],'distance_definition':DISTANCE_DEFINITION,
        'dataset_metadata':{'window_normalization':'global_rms'},
        'source_sha256':{'generator.py':'frozen-code'},'dataset_sha256':{'features':'frozen-data'}})
    fold=root/'subject_0'
    write_json(fold/'fold.json',{'subject_id':0,'status':'completed','target_class':1,
        'direction':'0_to_1','eligible_trial_ids':[0,1,2],'threshold':2.,'recognition':{}})
    probabilities=np.array([[.9,.1]]*4)
    write_json(fold/'observations.json',{'trial_ids':[0,1,2,3],'labels':[0,0,0,1],
                                       'probabilities':probabilities,'discrepancy':[1.,1.,1.,1.]})
    write_json(fold/'eligibility.json',eligibility_records(task='valence',direction='0_to_1',subject=0,
        trial_ids=[0,1,2,3],labels=[0,0,0,1],probabilities=probabilities))
    write_json(fold/'calibration/source_trials.json',{'subject_ids':[1,2],'trial_ids':[0,0],
                                                   'labels':[1,1],'discrepancy':[1.,2.]})
    for trial in (0,1):
        for arm,latent,edit,total in [('base',1.,3.,4.),('typicality',2.,4.,3.)]:
            attempt=fold/f'trial_{trial}'/arm/'attempt_0001'
            success=trial==0
            summary={
                'status':'completed','subject_id':0,'trial_id':trial,'objective':arm,
                'target_class':1,'direction':'0_to_1','checkpoint_sha256':'frozen-checkpoint',
                'report_output':'joint','decoded_distance_reference':'original_reconstruction',
                'd_z':latent,'decoder_latent_rmse':latent/10,
                'typicality':{'threshold':2.,'typical':True,'original_discrepancy':1.,
                              'counterfactual_discrepancy':1. if arm=='base' else .5},
                'latent_counterfactual':{'success':success,'predicted_class':1 if success else 0,
                                         'target_probability':.8 if success else .4},
                'required_target_probability':.5,'original':{'target_probability':.1},
                'decoded_trials':{'joint':{'original_reconstruction_mse':1.,'decoded_change_mse':edit**2,
                    'counterfactual_to_original_mse':total**2,'vcsc_original_reconstruction':3.,'vcsc_counterfactual':0.}},
                'physiology':{'all_required_passed':None,'passed_count':2,'available_count':4,
                              'required_count':5,'available_checks_passed':False},
                'physiological_tolerance':1e-8,'vcsc_original_input':2.,'selected_step':2,
                'steps_completed':3,'stop_reason':'max_steps'}
            write_json(attempt/'result.json',summary)
            write_json(attempt/'complete.json',{'sha256':{'result.json':file_sha256(attempt/'result.json')}})
    write_json(fold/'trial_2/base/attempt_0001/result.json',{'status':'error','error':'fixture error'})
    return root


def test_scalar_spaces_rmse_pairing_and_unsuccessful_endpoints(tmp_path):
    root=archive(tmp_path)
    out=tmp_path/'audit'
    report=build_audit([root],out,expected_subjects=1,figures=False)
    assert report['n_endpoints']==6 and report['n_unique_trials']==3
    assert not report['complete']  # one missing typicality endpoint
    assert report['validation']['passed']
    assert report['validation']['n_triangle_checked']==4
    assert report['validation']['n_triangle_violations']==0
    base=next(r for r in report['population'] if r['objective']=='base')
    assert base['d_z_median']==1 and base['decoder_latent_rmse_median']==.1
    assert base['delta_dec_median']==3 and base['e_cf_median']==4 and base['e_rec_median']==1
    assert base['n_eligible']==3 and base['d_z_n']==2 and base['n_error']==1
    assert base['n_unsuccessful_finite']==1
    import csv
    with (out/'paired_summary.csv').open() as f: pairs=list(csv.DictReader(f))
    change=next(r for r in pairs if r['objective']=='typicality' and r['metric']=='change_delta_dec')
    assert float(change['median'])==1 and int(change['n'])==2
    assert report['decoded_validity'].startswith('deferred')
    assert 'decoded_probability' not in (out/'trial_displacements.csv').read_text()
    assert 'N finite/eligible' in (out/'table2.tex').read_text()
    assert len(list(csv.DictReader((out/'exclusions.csv').open())))==1


def test_hash_tampering_is_rejected(tmp_path):
    root=archive(tmp_path)
    path=root/'subject_0/trial_0/base/attempt_0001/result.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='Saved artifact changed'):
        collect_audit([root])


def test_duplicate_roots_and_conflicting_protocols_are_rejected(tmp_path):
    root=archive(tmp_path)
    with pytest.raises(ValueError,match='Duplicate study roots'):
        collect_audit([root,root])
    other=tmp_path/'other'
    shutil.copytree(root,other)
    with pytest.raises(ValueError,match='Duplicate task/direction/subject'):
        collect_audit([root,other])
    path=other/'study.json'
    manifest=json.loads(path.read_text()); manifest['arguments']['typicality_quantile']=.1
    write_json(path,manifest)
    with pytest.raises(ValueError,match='Mixed experimental protocols'):
        collect_audit([root,other])


def test_triangle_violations_remain_visible_in_report(tmp_path):
    root=archive(tmp_path)
    path=root/'subject_0/trial_0/base/attempt_0001/result.json'
    summary=json.loads(path.read_text())
    summary['decoded_trials']['joint']['counterfactual_to_original_mse']=100.
    write_json(path,summary)
    write_json(path.with_name('complete.json'),{'sha256':{'result.json':file_sha256(path)}})
    report=build_audit([root],tmp_path/'audit',figures=False)
    assert report['validation']['n_triangle_violations']==1
    assert not report['validation']['passed']


def test_missing_distances_are_unavailable_not_zero(tmp_path):
    root=archive(tmp_path)
    path=root/'subject_0/trial_0/base/attempt_0001/result.json'
    summary=json.loads(path.read_text())
    del summary['decoded_trials']['joint']['counterfactual_to_original_mse']
    del summary['decoder_latent_rmse']
    write_json(path,summary)
    write_json(path.with_name('complete.json'),{'sha256':{'result.json':file_sha256(path)}})
    report=build_audit([root],tmp_path/'audit',figures=False)
    base=next(r for r in report['population'] if r['objective']=='base')
    assert base['e_cf_n']==1 and base['e_cf_median']==4
    assert base['decoder_latent_rmse_n']==1
    assert base['e_cf_exact_zero_n']==0
    assert not report['complete']
    assert report['validation']['n_missing_signal_triplets']==1
    summary=population_summary([{'status':'completed','e_cf':4.,'decoder_latent_rmse':.1},
                                {'status':'error'}])
    assert summary['e_cf_n']==1 and summary['decoder_latent_rmse_n']==1


def test_figures_and_tables_export_from_finite_failed_attempts(tmp_path):
    report=build_audit([archive(tmp_path)],tmp_path/'figures',figures=True)
    assert 'table2.png' in report['figures']
    assert len(report['figures'])==8
    assert all((tmp_path/'figures'/p).stat().st_size>100 for p in report['figures'])


def test_zero_eligible_subjects_keep_rows_for_only_the_declared_objectives(tmp_path):
    root=archive(tmp_path)
    manifest=json.loads((root/'study.json').read_text())
    manifest['folds'].append({'subject_id':1,'sha256':'other-frozen-checkpoint'})
    write_json(root/'study.json',manifest)
    fold=root/'subject_1'
    write_json(fold/'fold.json',{'subject_id':1,'status':'completed','target_class':1,
        'direction':'0_to_1','eligible_trial_ids':[],'threshold':2.,'recognition':{}})
    write_json(fold/'observations.json',{'trial_ids':[0],'labels':[1],
                                       'probabilities':[[.9,.1]],'discrepancy':[1.]})
    write_json(fold/'eligibility.json',eligibility_records(task='valence',direction='0_to_1',subject=1,
        trial_ids=[0],labels=[1],probabilities=[[.9,.1]]))
    out=tmp_path/'audit'
    report=build_audit([root],out,figures=False)
    import csv
    with (out/'subject_displacements.csv').open() as f:
        zero=[r for r in csv.DictReader(f) if r['subject_id']=='1']
    assert {r['objective'] for r in zero}=={'base','typicality'}
    assert all(r['n_eligible']=='0' and r['e_cf_n']=='0' and r['e_cf_median']=='' for r in zero)
    assert report['cohort'][0]['n_zero_eligible_subjects']==1


def test_negative_mse_is_rejected_instead_of_becoming_nan(tmp_path):
    root=archive(tmp_path)
    path=root/'subject_0/trial_0/base/attempt_0001/result.json'
    summary=json.loads(path.read_text()); summary['decoded_trials']['joint']['decoded_change_mse']=-1
    write_json(path,summary)
    write_json(path.with_name('complete.json'),{'sha256':{'result.json':file_sha256(path)}})
    with pytest.raises(ValueError,match='finite nonnegative'):
        collect_audit([root])
