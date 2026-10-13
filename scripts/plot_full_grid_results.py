"""Render the completed 3x3 distillation grid from its frozen measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

TEMPERATURES=(3,5,10)
ALPHAS=(.1,.3,.5)
SEEDS=(378,379,380)
PALETTE={'teacher':'#694483','baseline':'#939099','student':'#88994A','ink':'#40384C','grid':'#E4DFE9'}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def identifier(t,a):return f'T{t}-a{int(round(a*10)):02d}'

def collect(study,release):
    if study.get('status')!='completed' or study.get('grid_run_count')!=27:
        raise ValueError('All 27 grid runs must have completed')
    if release.get('status')!='completed' or not release.get('evaluation_does_not_select_models'):
        raise ValueError('A completed, selection-independent reference evaluation is required')
    runs={(int(r['seed']),r['id']):r for r in study['grid_runs']}
    wanted={(s,identifier(t,a)) for s in SEEDS for t in TEMPERATURES for a in ALPHAS}
    if len(study['grid_runs'])!=27 or set(runs)!=wanted:
        raise ValueError('The exact nine configurations and three seeds are required')
    locked=release['evaluations']['locked_holdout']
    validation,reference,rows=[],[],[]
    for t in TEMPERATURES:
        v_row,h_row=[],[]
        for a in ALPHAS:
            name=identifier(t,a)
            v,h=[],[]
            for seed in SEEDS:
                run=runs[(seed,name)]
                metric=locked[f'grid_{name}_seed_{seed}']
                if metric['model_sha256']!=run['model_sha256'] or metric['freeze_sha256']!=release['freeze_sha256']:
                    raise ValueError('Reference measurement is not bound to this frozen checkpoint')
                if metric['count']!=5850:
                    raise ValueError('Every checkpoint must use the same 5,850 images')
                v.append(float(run['validation_accuracy'])*100)
                h.append(float(metric['accuracy'])*100)
            row={'id':name,'temperature':t,'alpha':a,'validation_mean_percent':statistics.mean(v),'validation_sample_sd_pp':statistics.stdev(v),'reference_mean_percent':statistics.mean(h),'reference_sample_sd_pp':statistics.stdev(h),'seeds':list(SEEDS)}
            rows.append(row);v_row.append((row['validation_mean_percent'],row['validation_sample_sd_pp']));h_row.append((row['reference_mean_percent'],row['reference_sample_sd_pp']))
        validation.append(v_row);reference.append(h_row)
    if release['selection']['selected_candidate_id']!=study['selection']['selected_candidate_id']:
        raise ValueError('The evaluation and validation selection disagree')
    return np.array(validation),np.array(reference),rows

def render(study_path,release_path,output):
    study,release=read(study_path),read(release_path)
    validation,reference,rows=collect(study,release)
    selected=study['selection']['selected_candidate_id']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':PALETTE['ink'],'axes.labelcolor':PALETTE['ink'],'xtick.color':PALETTE['ink'],'ytick.color':PALETTE['ink'],'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(15,8),facecolor='#FCFBFD')
    fig.subplots_adjust(left=.09,right=.92,bottom=.27,top=.77,wspace=.31)
    fig.text(.06,.95,'Traffic sign distillation: complete temperature / alpha grid',fontsize=23,fontweight='bold',color=PALETTE['teacher'])
    fig.text(.06,.90,'9 configurations | 3 matched development seeds each | identical 34,406-parameter students',fontsize=13)
    cmap=LinearSegmentedColormap.from_list('distillation_olive',['#F1F3E8','#CDD7A4','#8A9C4B','#566A2D'])
    both=np.concatenate([validation[:,:,0].flatten(),reference[:,:,0].flatten()])
    norm=Normalize(vmin=np.floor(both.min()),vmax=np.ceil(both.max()))
    for axis,values,title in zip(axes,(validation,reference),('Validation accuracy: selection data','Reference accuracy: previously reported holdout')):
        heat=axis.imshow(values[:,:,0],cmap=cmap,norm=norm,aspect='equal')
        axis.set_title(title,fontsize=14,fontweight='bold',pad=18)
        axis.set_xticks(range(3),[f'alpha {a:.1f}' for a in ALPHAS])
        axis.set_yticks(range(3),[f'T = {t}' for t in TEMPERATURES])
        axis.set_xlabel('Hard-label loss weight',labelpad=13)
        axis.set_ylabel('Distillation temperature',labelpad=13)
        axis.set_xticks(np.arange(-.5,3,1),minor=True);axis.set_yticks(np.arange(-.5,3,1),minor=True)
        axis.grid(which='minor',color='#FFFFFF',linewidth=3);axis.tick_params(which='minor',bottom=False,left=False)
        for i,t in enumerate(TEMPERATURES):
            for j,a in enumerate(ALPHAS):
                mean,sd=values[i,j]
                color='#FFFFFF' if norm(mean)>.66 else PALETTE['ink']
                axis.text(j,i,f'{mean:.2f}%\n± {sd:.2f} pp',ha='center',va='center',color=color,fontsize=16,fontweight='bold')
                if identifier(t,a)==selected:
                    axis.add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor=PALETTE['teacher'],linewidth=3))
    bar=fig.colorbar(heat,ax=axes,location='right',fraction=.025,pad=.035)
    bar.set_label('Accuracy (%)')
    fig.text(.06,.12,f'Purple border: validation-selected setting {selected}. Every cell shows mean ± sample SD over seeds 378, 379 and 380.',fontsize=12)
    fig.text(.06,.075,'All three seeds participate in development selection. Reference outcomes do not choose the configuration or deployment seed.',fontsize=12)
    fig.text(.06,.03,'The reference images had published results before this extension; these are follow-up comparisons, not an untouched test.',fontsize=11,color='#70677B')
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(output/f'full-grid-comparison.{suffix}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt={'status':'rendered_from_completed_measurements','selected_candidate_id':selected,'grid_cells':9,'runs':27,'rows':rows,'study_sha256':hashlib.sha256(study_path.read_bytes()).hexdigest(),'release_sha256':hashlib.sha256(release_path.read_bytes()).hexdigest(),'palette':PALETTE}
    (output/'full-grid-figure-data.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'grid_cells':9,'runs':27,'selected_candidate_id':selected}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.study,a.release,a.output)

if __name__=='__main__':main()
