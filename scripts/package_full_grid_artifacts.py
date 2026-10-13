"""Package the measured full-grid models and immutable starting evidence.

Both archives use the existing hash-pinned download descriptor format. The
starting-evidence archive permits Kaggle reproduction without a private kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def filename(role):
    if role=='teacher':return 'teacher.keras'
    if role=='baseline_seed_378':return 'baseline.keras'
    if role.startswith('baseline_seed_'):return role.replace('_','-')+'.keras'
    if role=='deployment_student':return 'student.keras'
    if role=='deployment_fp32':return 'student-fp32.tflite'
    if role=='deployment_int8':return 'student-int8.tflite'
    if role.startswith('grid_') or role.startswith('control_'):
        group,tail=role.split('_',1)
        return group+'/'+tail.replace('_','-')+'.keras'
    raise ValueError('Unknown frozen model role: '+role)

def write_archive(path,payload):
    if path.exists():raise FileExistsError('Preserve existing artifact packages')
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted(payload.items()):
            if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:
                raise ValueError('Archive paths must be portable and relative')
            member=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            member.create_system=3;member.external_attr=0o100644<<16;member.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(member,raw)
    if path.stat().st_size>25*1024*1024:
        raise ValueError('Artifact exceeds the site per-file limit; use verified parts instead')
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist())==set(payload)
        for name,raw in payload.items():
            if archive.read(name)!=raw:raise ValueError('Archived artifact differs from its frozen bytes')
    return [{'path':name,'bytes':len(raw),'sha256':sha(raw)} for name,raw in sorted(payload.items())]

def package(run,release,base_run,output,url_prefix,descriptor_path):
    sys.path.insert(0,str(ROOT))
    from scripts.full_grid_study import inspect_base_run,load_plan
    from knowledge_distillation.evaluation import validate_freeze
    plan=load_plan(ROOT/'configs/full-grid-plan.json')
    base=inspect_base_run(base_run,plan)
    summary=read(release/'release-summary.json')
    freeze=validate_freeze(release/'evaluation-freeze.json',run/'splits.json',source_root=ROOT)
    if summary.get('status')!='completed' or summary['freeze_sha256']!=freeze['freeze_sha256']:
        raise ValueError('The completed release summary and freeze must match')
    if set(summary['models'])!=set(freeze['models']) or set(summary['evaluations']['locked_holdout'])!=set(freeze['models']):
        raise ValueError('Every frozen model must have matching evaluation and delivery records')
    available={}
    for root in (run,release/'runtime-models'):
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.keras','.tflite'):
                raw=path.read_bytes();available[(len(raw),sha(raw))]=raw
    payload={};roles={}
    for role,binding in sorted(freeze['models'].items()):
        declared=summary['models'][role]
        if binding['sha256']!=declared['sha256'] or binding['bytes']!=declared['bytes']:
            raise ValueError('Measured and frozen model inventories differ')
        metric=summary['evaluations']['locked_holdout'][role]
        if metric['model_sha256']!=binding['sha256'] or metric['freeze_sha256']!=freeze['freeze_sha256'] or metric['count']!=5850:
            raise ValueError('A packaged model lacks the same frozen 5,850-image evaluation')
        raw=available.get((binding['bytes'],binding['sha256']))
        if raw is None:raise ValueError('A measured checkpoint is missing: '+role)
        name=filename(role);payload[name]=raw;roles[role]=name
    if len([role for role in roles if role.startswith('grid_')])!=27 or len([role for role in roles if role.startswith('control_')])!=3:
        raise ValueError('Every grid and diagnostic checkpoint must be packaged')
    source_payload={}
    for relative,binding in base['inventory'].items():
        path=base_run/relative
        raw=path.read_bytes()
        if len(raw)!=binding['bytes'] or sha(raw)!=binding['sha256']:
            raise ValueError('Starting evidence changed after inspection')
        source_payload[relative]=raw
    archives=[]
    for name,path_key,content in [('knowledge-distillation-full-grid-models.zip','models',payload),('knowledge-distillation-full-grid-base-evidence.zip','original-base-run',source_payload)]:
        path=output/name
        inventory=write_archive(path,content)
        raw=path.read_bytes()
        archives.append({'name':'full-grid-models' if path_key=='models' else 'original-base-run','kind':'zip','path':path_key,'url':url_prefix.rstrip('/')+'/'+name,'bytes':len(raw),'sha256':sha(raw),'inventory':inventory})
    descriptor={'schema_version':1,'release_freeze_sha256':summary['freeze_sha256'],'base_inventory_sha256':plan['base_bindings']['artifact_inventory_sha256'],'model_roles':roles,'artifacts':archives}
    descriptor_path.parent.mkdir(parents=True,exist_ok=True)
    if descriptor_path.exists():raise FileExistsError('Use a new descriptor path')
    descriptor_path.write_text(json.dumps(descriptor,indent=2)+'\n',encoding='utf-8')
    base_artifact=archives[1]
    base_descriptor={'schema_version':1,'kind':'full-grid-base-archive','url':base_artifact['url'],
        'size_bytes':base_artifact['bytes'],'sha256':base_artifact['sha256'],'archive_prefix':'',
        'inventory':base['inventory'],'inventory_sha256':plan['base_bindings']['artifact_inventory_sha256']}
    (output/'base-download.json').write_text(json.dumps(base_descriptor,indent=2)+'\n',encoding='utf-8')
    receipt={'status':'passed_frozen_model_packaging','models':len(roles),'original_base_files':len(source_payload),'artifacts':[{key:value for key,value in artifact.items() if key!='inventory'} for artifact in archives]}
    (output/'package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)
    return descriptor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','release','base-run','output','descriptor'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--url-prefix',default='https://park-dongmyung.com/downloads')
    a=p.parse_args()
    if not a.url_prefix.startswith('https://'):raise ValueError('Public artifacts require HTTPS')
    package(a.run,a.release,a.base_run,a.output,a.url_prefix,a.descriptor)

if __name__=='__main__':main()
