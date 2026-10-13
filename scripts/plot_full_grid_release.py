"""Map a completed full-grid release to the existing measured-cost figure.

Aliases are bound to actual frozen model roles; the original summary is retained.
This presentation projection never selects a candidate from reference accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_results import make_plot

def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def project(summary,freeze):
    if (summary.get('status')!='completed'
        or summary.get('evaluation_does_not_select_models') is not True
        or freeze.get('freeze_sha256')!=canonical({k:v for k,v in freeze.items() if k!='freeze_sha256'})
        or summary['freeze_sha256']!=freeze['freeze_sha256']):
        raise ValueError('Require a completed release bound to its immutable freeze')
    candidate=summary['selection']['selected_candidate_id']
    selected=freeze['config']['deployment_candidate_id']
    if candidate!=selected or summary['deployment']['seed']!=378:
        raise ValueError('The figure must follow the frozen validation selection')
    held=summary['evaluations']['locked_holdout']
    aliases={'teacher':'teacher','baseline':'baseline_seed_378',
             'distilled':'deployment_student','fp32':'deployment_fp32','int8':'deployment_int8'}
    for seed in (379,380):
        aliases['baseline_seed_'+str(seed)]='baseline_seed_'+str(seed)
        aliases['distilled_seed_'+str(seed)]=f'grid_{candidate}_seed_{seed}'
    metrics={}
    for alias,role in aliases.items():
        binding=freeze['models'][role]
        metric=held[role]
        if (metric['model_sha256']!=binding['sha256']
            or metric['freeze_sha256']!=freeze['freeze_sha256'] or metric['count']!=5850):
            raise ValueError('A figure metric differs from the frozen evaluated model')
        metrics[alias]=metric
    if sum(role.startswith('grid_') for role in freeze['models'])!=27:
        raise ValueError('The release must contain the complete nine-condition study')
    return {**summary,'evaluations':{**summary['evaluations'],'locked_holdout':metrics},
            '_figure_role_aliases':aliases,
            '_scope':'Three development seeds from the full grid; already reported reference data.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','freeze','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    raw=args.summary.read_bytes()
    projection=project(json.loads(raw),json.loads(args.freeze.read_text(encoding='utf-8')))
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'release-figure-projection.json'
    if path.exists():raise FileExistsError('Preserve completed figure evidence')
    path.write_text(json.dumps(projection,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    receipt=make_plot(projection,args.output,'knowledge-distillation-release')
    receipt['png']=Path(receipt['png']).name
    receipt['svg']=Path(receipt['svg']).name
    receipt.update(parent_summary_sha256=hashlib.sha256(raw).hexdigest(),
                   freeze_sha256=projection['freeze_sha256'],
                   aliases=projection['_figure_role_aliases'],
                   development_seeds_not_independent_confirmation=True)
    (args.output/'release-figure-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
