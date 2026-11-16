"""Retrain NH/family heads and gate every old class on unchanged control groups."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from scripts.train_weather_round4_specialist import load_vectors,count
from scripts.train_mvp import train
from scripts.train_machinery_hierarchy import calibrate
from src.config.machinery_taxonomy import family_of
from src.models.hierarchy import predict_hierarchy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'

def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def evaluate(bundle,rows,vectors,markings=None):
    rr=[]
    for r in rows:
        pred=predict_hierarchy(bundle,vectors[r['key']],markings.get(r['sha256']) if markings else None)
        rr.append(dict(key=r['key'],expected=r.get('model'),expected_family=r.get('expected_family',family_of(r.get('model'))),candidate=pred['family_candidate'],family=pred['family'],label=pred['label_candidate']))
    return dict(**count(rr),by_class={str(label):count([r for r in rr if str(r['expected'])==label]) for label in sorted({str(r['expected']) for r in rr})},rows=rr)

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes());baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes())
    original=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes());old_keys={r['key'] for r in original['train']+original['family_only_train']}
    full,crop=load_vectors();z=np.load(OUT/'features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    rr=data['train']+data['family_only_train']+[r for s in ('test','winter_test','fresh_test','new_photos_test','round4_test') for r in data[s]]
    vv={r['key']:np.concatenate((full[r['key']],crop[r['key']])) for r in rr}
    families=[dict(r,model=family_of(r['model'])) for r in data['train']]+[dict(r,model=r['family']) for r in data['family_only_train']]
    known=[r for r in families if r['key'] in old_keys or r['model']!='unknown']
    family_head,ft=train(known,vv,tune=False,min_groups=10)
    nh_rows=[dict(r,model='nh_t5000_family' if r['model']=='nh_t5000_family' else 'not_nh') for r in families]
    nh_head,nt=train(nh_rows,vv,min_groups=10)
    fc=calibrate(ft['oof_records'],family_head['classes'],min_groups=10,target_precision=.98)
    nc=calibrate(nt['oof_records'],nh_head['classes'],min_groups=10,target_precision=.98)
    for c,floor in ((fc,.90),(nc,.95)):
        for value in c.values():
            if value['threshold'] is not None:value['threshold']=max(floor,value['threshold'])
    variants={'baseline':baseline}
    for name in ('family_only','nh_only','combined'):
        b=copy.deepcopy(baseline)
        if name!='nh_only':b['heads']['family']=family_head;b['calibration']['family']=fc
        if name!='family_only':b['heads']['nh_specialist']=nh_head;b['calibration']['nh_specialist']=nc
        b['manifest_sha256']=hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest()
        variants[name]=b;save(name+'_candidate.json',b)
    comparisons={name:{s:evaluate(b,data[s],vv) for s in ('test','winter_test','fresh_test','new_photos_test','round4_test')} for name,b in variants.items()}
    gates={}
    for name in variants:
        if name=='baseline':continue
        current=comparisons[name]['test'];old=comparisons['baseline']['test']
        class_gates={label:dict(family=value['family_correct']>=old['by_class'][label]['family_correct'],
            label=value['label_correct']>=old['by_class'][label]['label_correct'],correct_answers=value['correct_answers']>=old['by_class'][label]['correct_answers'],
            wrong_answers=value['wrong_answers']<=old['by_class'][label]['wrong_answers']) for label,value in current['by_class'].items()}
        controls={s:dict(family=comparisons[name][s]['family_correct']>=comparisons['baseline'][s]['family_correct'],wrong=comparisons[name][s]['wrong_answers']<=comparisons['baseline'][s]['wrong_answers']) for s in ('winter_test','fresh_test','new_photos_test','round4_test')}
        gates[name]=dict(classes=class_gates,controls=controls,eligible=all(all(v.values()) for v in list(class_gates.values())+list(controls.values())))
    save('recognition_comparison.json',comparisons);save('recognition_gates.json',gates);save('recognition_training.json',dict(family=ft,nh_specialist=nt))
    print({name:{s:{k:r[k] for k in ('family_correct','answers','wrong_answers','correct_answers')} for s,r in splits.items()} for name,splits in comparisons.items()},flush=True)
    print('eligible',{k:v['eligible'] for k,v in gates.items()},flush=True)
    print('failed classes',{name:{label:checks for label,checks in value['classes'].items() if not all(checks.values())} for name,value in gates.items()},flush=True)

if __name__=='__main__':main()
