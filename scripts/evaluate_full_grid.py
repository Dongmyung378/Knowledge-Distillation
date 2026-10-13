"""Evaluate every frozen full-grid checkpoint and measure its deployment costs.

This CPU-only stage never trains or selects candidates from reference outcomes.
Run after scripts/full_grid_study.py --stage freeze has completed on Kaggle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def bound_selection(path,frozen):
    raw=path.read_bytes()
    binding=frozen['config']['selection_report_file']
    if len(raw)!=binding['bytes'] or hashlib.sha256(raw).hexdigest()!=binding['sha256']:
        raise ValueError('Selection report differs from the frozen validation decision')
    selection=json.loads(raw)
    config=frozen['config']
    if (selection['selected_candidate_id']!=config['selected_candidate_id']
        or selection['selected_configuration']!=config['selected_configuration']
        or selection['deployment']['candidate_id']!=config['selected_candidate_id']
        or selection['deployment']['seed']!=config['deployment_seed']
        or selection['deployment']['selected_on']!='validation'
        or selection['final_outcomes_used_for_selection'] is not False):
        raise ValueError('Deployment must follow the exact frozen validation selection')
    return selection

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Validation-only selection report; defaults to run/selection-report.json')
    args=parser.parse_args()
    for key in ('TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    sys.path.insert(0,str(ROOT))
    from knowledge_distillation.runtime import configure_cpu, file_sha256
    configure_cpu(1)
    from knowledge_distillation.data import load_manifest
    from knowledge_distillation.evaluation import evaluate_keras, evaluate_tflite, freeze_evaluation, validate_freeze, build_official_test_manifest
    from knowledge_distillation.export import export_student, assess_int8_acceptance
    from knowledge_distillation.benchmark import benchmark_models

    run, output=args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new evaluation directory; never overwrite published measurements')
    output.mkdir(parents=True)
    manifest=load_manifest(run/'splits.json')
    training_freeze=run/'evaluation-freeze.json'
    frozen=validate_freeze(training_freeze,manifest,source_root=ROOT)
    selection=bound_selection(args.selection or run/'selection-report.json',frozen)
    # Explicit deployment bindings keep reference-test outcomes out of selection.
    deployment=selection['deployment']
    candidate_id=deployment['candidate_id']
    deployment_seed=int(deployment['seed'])
    assert deployment_seed==378
    models={}
    available={}
    for path in run.rglob('*.keras'):
        available.setdefault((path.stat().st_size,file_sha256(path)),path)
    for role,binding in sorted(frozen['models'].items()):
        model=available.get((binding['bytes'],binding['sha256']))
        if model is None:
            raise ValueError('Frozen checkpoint is missing: '+role)
        models[role]=model
    selected=(run/'students'/f'seed-{deployment_seed}'/(candidate_id or 'baseline')/'student.keras')
    assert selected.is_file()
    validate_freeze(training_freeze,manifest,selected,source_root=ROOT)
    expected=frozen['models'][frozen['config']['deployment_model_name']]
    if selected.stat().st_size!=expected['bytes'] or file_sha256(selected)!=expected['sha256']:
        raise ValueError('Deployment checkpoint differs from the frozen selected model')
    runtime=output/'runtime-models'
    runtime.mkdir()
    student=runtime/'student.keras'
    shutil.copyfile(selected,student)
    # Baseline and teacher reference models are bound by the same training freeze.
    reference={
        'teacher':run/'teachers/variant-0/teacher.keras',
        'baseline':run/'students/seed-378/baseline/student.keras',
        'distilled':student,
    }
    for model in reference.values():validate_freeze(training_freeze,manifest,model,source_root=ROOT)
    print('FULL GRID EXPORT: training-only calibration',flush=True)
    export=export_student(student,manifest,args.data,output/'export',calibration_count=256,seed=378,freeze_path=training_freeze)
    for role,filename in [('fp32','student-fp32.tflite'),('int8','student-int8.tflite')]:
        target=runtime/filename
        shutil.copyfile(output/'export'/filename,target)
        reference[role]=target
    validation={role:evaluate_tflite(reference[role],manifest,args.data,'validation',output/'validation'/role) for role in ('fp32','int8')}
    benchmark=benchmark_models(reference,manifest,args.data,output/'cpu-benchmark',sample_count=128,warmup=10,repetitions=3,threads=1)
    acceptance=assess_int8_acceptance(validation['fp32'],validation['int8'],benchmark['models']['fp32'],benchmark['models']['int8'])
    save(acceptance,output/'int8-acceptance.json')
    config={
        'study':'full-temperature-alpha-grid-three-seed-extension',
        'training_freeze_sha256':frozen['freeze_sha256'],
        'selection':selection,
        'deployment_seed':deployment_seed,
        'deployment_candidate_id':candidate_id,
        'deployment_format':acceptance['selected_format'],
        'int8_acceptance':acceptance,
        'deployment_selection_uses_reference_outcomes':False,
        'reference_evaluation_exposure':'The same holdout already had published outcomes under the previous four-candidate study. This extension is not an untouched independent test.',
        'export_report_sha256':file_sha256(output/'export/export-report.json'),
        'cpu_benchmark_sha256':file_sha256(output/'cpu-benchmark/benchmark.json'),
        'evaluator_sha256':file_sha256(Path(__file__)),
        'training_config':frozen['config'],
    }
    release_models={**models,'deployment_student':student,'deployment_fp32':reference['fp32'],'deployment_int8':reference['int8']}
    freeze_path=output/'evaluation-freeze.json'
    release_freeze=freeze_evaluation(release_models,manifest,config,freeze_path,source_root=ROOT)
    print('FULL GRID DEPLOYMENT FROZEN',release_freeze['freeze_sha256'],flush=True)
    metrics={}
    for index,(role,model) in enumerate(sorted(release_models.items()),1):
        print(f'FULL GRID REFERENCE {index}/{len(release_models)} {role}',flush=True)
        evaluator=evaluate_tflite if model.suffix=='.tflite' else evaluate_keras
        metrics[role]=evaluator(model,manifest,args.data,'locked_holdout',output/'evaluations/locked_holdout'/role,freeze_path=freeze_path)
        save({'phase':'reference_evaluation','completed':index,'total':len(release_models),'last_role':role},output/'progress.json')
    print('FULL GRID HISTORICAL OFFICIAL AUDIT',flush=True)
    official=build_official_test_manifest(args.data,manifest,freeze_path)
    save(official,output/'official-test-manifest.json')
    audit={}
    for role in ('teacher','baseline','distilled','fp32','int8'):
        model=reference[role]
        evaluator=evaluate_tflite if model.suffix=='.tflite' else evaluate_keras
        audit[role]=evaluator(model,official,args.data,'official_test',output/'evaluations/official_test'/role,freeze_path=freeze_path)
    summary={
        'schema_version':1,'status':'completed','study':config['study'],
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'training_location':'Kaggle GPU only','reference_evaluation_device':'CPU',
        'selection':selection,'deployment':{**deployment,'format':acceptance['selected_format'],'int8_acceptance':acceptance},
        'evaluations':{'locked_holdout':metrics,'official_test':audit},
        'benchmark':benchmark,'export':export,
        'training_freeze_sha256':frozen['freeze_sha256'],'freeze_sha256':release_freeze['freeze_sha256'],
        'manifest_sha256':manifest['manifest_sha256'],'counts':manifest['counts'],
        'exposure_statement':config['reference_evaluation_exposure'],
        'evaluation_does_not_select_models':True,
        'models':{role:{'path':str(model.relative_to(run)) if model.is_relative_to(run) else str(model.relative_to(output)),'sha256':file_sha256(model),'bytes':model.stat().st_size} for role,model in release_models.items()},
    }
    save(summary,output/'release-summary.json')
    print(json.dumps({'status':'completed','models_evaluated':len(metrics),'deployment_candidate':candidate_id,'deployment_format':acceptance['selected_format']},sort_keys=True),flush=True)

if __name__=='__main__':main()
