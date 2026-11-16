"""Train a positive MTZ-family specialist without moving existing family boundaries."""
import json,copy
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from scripts.train_winter_round2 import vectors
from scripts.train_mvp import fit
from scripts.train_machinery_hierarchy import serialize,calibrate
from src.config.machinery_taxonomy import family_of
from src.models.hierarchy import predict_hierarchy,vote,qualified
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'

def load_vectors():
    full,crop=vectors();z=np.load(ROOT/'output/robustness_round3/features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    return full,crop

def count(rr):
    return dict(total=len(rr),family_correct=sum(r['candidate']==r['expected_family'] for r in rr),
        answers=sum(r['family'] is not None for r in rr),wrong_answers=sum(r['family'] is not None and r['family']!=r['expected_family'] for r in rr),
        correct_answers=sum(r['family']==r['expected_family'] for r in rr),
        label_correct=sum(r['label']==r['expected'] for r in rr if r['expected'] not in (None,'other')))

def proposal(bundle,x,head,thresholds,mode):
    result=predict_hierarchy(bundle,x)
    if head is None:return result
    label,score,_=vote(head,x)
    positive=label=='mtz_80_family' and qualified(thresholds,label,score)
    nhlabel,nhscore,_=vote(bundle['heads']['nh_specialist'],x)
    nh=nhlabel=='nh_t5000_family' and qualified(bundle['calibration']['nh_specialist'],nhlabel,nhscore)
    if positive and not nh:
        result['family_candidate']='mtz_80_family'
        result['label_candidate']=vote(bundle['heads']['mtz_specialist'],x)[0]
        vehicle=vote(bundle['heads']['type'],x)[0]
        result['family']='mtz_80_family' if vehicle=='tractor' else None
    elif positive and nh:
        result['family']=None
    if mode=='veto' and label=='not_mtz80' and score>=.99 and qualified(thresholds,label,score) and result['family_candidate']=='mtz_80_family':
        result['family']=None
    return result

def main():
    data=json.loads((ROOT/'output/robustness_round3/dataset_manifest.json').read_bytes())
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes());full,crop=load_vectors()
    originals=data['train']+data['family_only_train'];rr=[dict(r,model=family_of(r['model'])) for r in data['train']]+[dict(r,model=r['family']) for r in data['family_only_train']]
    rows=[dict(r,model='mtz_80_family' if r['model']=='mtz_80_family' else 'not_mtz80') for r in rr]
    x=np.stack([np.concatenate((full[r['key']],crop[r['key']])) for r in rows]);y=np.array([r['model'] for r in rows]);groups=np.array([r['group'] for r in rows]);classes=np.unique(y)
    zz=np.load(ROOT/'output/winter_round2/gray_features.npz');gv=dict(zip(zz['keys'],zz['values'],strict=True))
    gray={r['key']:np.concatenate((gv['full/'+r['key']],gv['crop/'+r['key']])) for r in rows if 'full/'+r['key'] in gv}
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=911).split(x,y,groups));trials=[]
    for aug in (False,True):
        for c in (.03,.1,.3,1.,3.):
            p=np.zeros((len(rows),len(classes)))
            for a,b in folds:
                assert not set(groups[a]) & set(groups[b])
                train_rows=[rows[i] for i in a];xx=x[a]
                aa=[i for i in a if rows[i]['key'] in gray] if aug else []
                if aa:
                    train_rows += [dict(rows[i],key=rows[i]['key']+'/gray') for i in aa]
                    xx=np.concatenate((xx,np.stack([gray[rows[i]['key']] for i in aa])))
                model=fit(train_rows,xx,c*len(a)/len(train_rows));pp=model.predict_proba(x[b])
                for j,label in enumerate(model.classes_):p[b,list(classes).index(label)]=pp[:,j]
            score=float(balanced_accuracy_score(y,classes[p.argmax(1)]));trials.append((score,aug,c,p));print('trial',aug,c,score,flush=True)
    score,aug,c,p=max(trials,key=lambda t:(t[0],not t[1],-t[2]));extra=[r for r in rows if r['key'] in gray] if aug else []
    train_rows=rows+[dict(r,key=r['key']+'/gray') for r in extra];xx=np.concatenate((x,np.stack([gray[r['key']] for r in extra]))) if extra else x
    model=fit(train_rows,xx,c*len(rows)/len(train_rows));head=serialize(model)
    records=[dict(group=r['group'],expected=r['model'],probabilities=pp.tolist()) for r,pp in zip(rows,p,strict=True)]
    thresholds=calibrate(records,classes,min_groups=10,target_precision=.98)
    for v in thresholds.values():
        if v['threshold'] is not None:v['threshold']=max(.95,v['threshold'])
    variants={}
    for mode in ('baseline','positive','veto'):
        variants[mode]={}
        for split in ('test','winter_test','fresh_test','new_photos_test'):
            result=[]
            for r in data[split]:
                xx=np.concatenate((full[r['key']],crop[r['key']]))
                pred=proposal(baseline,xx,None if mode=='baseline' else head,thresholds,mode)
                result.append(dict(key=r['key'],expected=r.get('model'),expected_family=r.get('expected_family',family_of(r.get('model'))),candidate=pred['family_candidate'],family=pred['family'],label=pred['label_candidate']))
            variants[mode][split]=dict(**count(result),by_class={str(label):count([r for r in result if str(r['expected'])==label]) for label in sorted({str(r['expected']) for r in result})},rows=result)
    for name,value in (('mtz_family_head.json',dict(head=head,calibration=thresholds,training=dict(c=c,grayscale=aug,oof_balanced_accuracy=score,trials=[dict(score=t[0],grayscale=t[1],c=t[2]) for t in trials],records=records))),('specialist_comparison.json',variants)):
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print('selected',aug,c,thresholds,flush=True)
    print({n:{s:{k:v[k] for k in ('family_correct','answers','wrong_answers','correct_answers')} for s,v in ss.items()} for n,ss in variants.items()},flush=True)

if __name__=='__main__':main()
