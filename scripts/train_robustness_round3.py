"""Compare an isolated T5000 specialist with a joint family update on frozen controls."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from scripts.train_winter_round2 import vectors,benchmark
from scripts.train_mvp import train
from scripts.train_machinery_hierarchy import calibrate
from src.config.machinery_taxonomy import family_of
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/robustness_round3'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    original=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes())
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes())
    full,crop=vectors();z=np.load(OUT/'features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    all_rows=data['train']+data['family_only_train']+data['test']+data['winter_test']+data['fresh_test']+data['new_photos_test']
    vv={r['key']:np.concatenate((full[r['key']],crop[r['key']])) for r in all_rows}
    families=[dict(r,model=family_of(r['model'])) for r in data['train']]+[dict(r,model=r['family']) for r in data['family_only_train']]
    old_keys={r['key'] for r in original['train']+original['family_only_train']}
    known_rows=[r for r in families if r['key'] in old_keys or r['model']!='unknown']
    joint=copy.deepcopy(baseline)
    head,summary=train(known_rows,vv,tune=False,min_groups=10)
    joint['heads']['family']=head
    joint['calibration']['family']=calibrate(summary['oof_records'],head['classes'],min_groups=10,target_precision=.98)
    for value in joint['calibration']['family'].values():
        if value['threshold'] is not None:value['threshold']=max(.90,value['threshold'])
    specialist=copy.deepcopy(baseline)
    nh_rows=[dict(r,model='nh_t5000_family' if r['model']=='nh_t5000_family' else 'not_nh') for r in families]
    nh_head,nh_training=train(nh_rows,vv,min_groups=10)
    specialist['heads']['nh_specialist']=nh_head
    specialist['calibration']['nh_specialist']=calibrate(nh_training['oof_records'],nh_head['classes'],min_groups=10,target_precision=.98)
    for v in specialist['calibration']['nh_specialist'].values():
        if v['threshold'] is not None:v['threshold']=max(.95,v['threshold'])
    # Only a calibrated positive specialist can change a family; negative votes cannot shift known boundaries.
    combined=copy.deepcopy(joint)
    combined['heads']['nh_specialist']=nh_head
    combined['calibration']['nh_specialist']=copy.deepcopy(specialist['calibration']['nh_specialist'])
    variants={'baseline':baseline,'known_additions':joint,'nh_specialist':specialist,'combined':combined}
    comparisons={name:{s:benchmark(b,data[s],vv) for s in ('test','winter_test','fresh_test','new_photos_test')} for name,b in variants.items()}
    for name,b in variants.items():
        if name!='baseline':
            b['manifest_sha256']=hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest()
            save(name+'_candidate.json',b)
    gates={}
    for name in ('known_additions','nh_specialist','combined'):
        cc=comparisons[name];bb=comparisons['baseline']
        gates[name]=dict(old_family=cc['test']['family_correct']>=bb['test']['family_correct'],
            old_wrong=cc['test']['wrong_answers']<=bb['test']['wrong_answers'],
            old_labels=cc['test']['label_correct']>=bb['test']['label_correct'],
            old_coverage=cc['test']['answers']>=bb['test']['answers'],
            winter=cc['winter_test']['family_correct']>=bb['winter_test']['family_correct'],
            fresh=cc['fresh_test']['wrong_answers']<=bb['fresh_test']['wrong_answers'],
            new_photos=cc['new_photos_test']['wrong_answers']<=bb['new_photos_test']['wrong_answers'])
    save('recognition_comparison.json',comparisons);save('recognition_training.json',dict(known_additions=summary,nh_specialist=nh_training));save('recognition_gates.json',gates)
    print(json.dumps({n:{s:{k:v[k] for k in ('total','family_correct','answers','wrong_answers','label_correct')} for s,v in ss.items()} for n,ss in comparisons.items()}),flush=True)
    print('Gates',gates,'NH calibration',specialist['calibration']['nh_specialist'],flush=True)

if __name__=='__main__':main()
