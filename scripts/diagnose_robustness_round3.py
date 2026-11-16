"""Controlled decomposition of known additions, unknown additions, and refusal gate."""
import copy
import json
import numpy as np
from pathlib import Path
from scripts.train_winter_round2 import vectors,benchmark
from scripts.train_mvp import train
from src.config.machinery_taxonomy import family_of
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/robustness_round3'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    data=json.loads((ROOT/'output/winter_round2/dataset_manifest.json').read_bytes())
    old=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes())
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes())
    full,crop=vectors()
    rows=data['train']+data['family_only_train']+data['test']+data['winter_test']+data['fresh_test']
    x={r['key']:np.concatenate((full[r['key']],crop[r['key']])) for r in rows}
    original={r['key'] for r in old['train']+old['family_only_train']}
    families=[dict(r,model=family_of(r['model'])) for r in data['train']]+[dict(r,model=r['family']) for r in data['family_only_train']]
    variants={}
    for name,rr in (
        ('known_only',[r for r in families if r['key'] in original or r['model']!='unknown']),
        ('unknown_only',[r for r in families if r['key'] in original or r['model']=='unknown']),
    ):
        candidate=copy.deepcopy(baseline)
        head,summary=train(rr,x,tune=False,min_groups=10)
        candidate['heads']['family']=head
        candidate['calibration']['family']=copy.deepcopy(summary['thresholds'])
        for v in candidate['calibration']['family'].values():
            if v['threshold'] is not None:v['threshold']=max(.9,v['threshold'])
        variants[name]={s:benchmark(candidate,data[s],x) for s in ('test','winter_test','fresh_test')}
        save(name+'_bundle.json',candidate)
        save(name+'_training.json',summary)
        print(name,{s:{k:v[k] for k in ('family_correct','answers','wrong_answers')} for s,v in variants[name].items()},flush=True)
    gate_rows=[dict(r,model='unknown' if r['model']=='unknown' else 'known') for r in families]
    gate,summary=train(gate_rows,x,min_groups=10)
    candidate=copy.deepcopy(baseline)
    candidate['heads']['unknown_gate']=gate
    candidate['calibration']['unknown_gate']=summary['thresholds']
    for v in candidate['calibration']['unknown_gate'].values():
        if v['threshold'] is not None:v['threshold']=max(.95,v['threshold'])
    variants['frozen_family_gate']={s:benchmark(candidate,data[s],x) for s in ('test','winter_test','fresh_test')}
    save('gate_bundle.json',candidate)
    save('gate_training.json',summary)
    save('regression_ablation.json',variants)
    print('gate',{s:{k:v[k] for k in ('family_correct','answers','wrong_answers')} for s,v in variants['frozen_family_gate'].items()},flush=True)

if __name__=='__main__':main()
