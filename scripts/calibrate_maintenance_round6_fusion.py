"""Calibrate balanced retrieval and discriminative confirmation on grouped OOF predictions."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.train_trailer_round5_verifier import vectors
from src.models.hierarchy import vote,qualified
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def balance(value):
    parts=[value[...,:384],value[...,384:]]
    result=np.concatenate([v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9) for v in parts],axis=-1)
    return result/np.maximum(np.linalg.norm(result,axis=-1,keepdims=True),1e-9)
def choose(scores,labels,groups,floor,min_groups=5):
    choices=[]
    for t in sorted({floor}|{float(v) for v in scores if np.isfinite(v) and v>=floor}):
        selected=scores>=t;gg=set(groups[selected]);cc={g for g in gg if all(labels[i]=='attachment' for i in np.flatnonzero(selected&(groups==g)))}
        if len(gg)>=min_groups and len(cc)/len(gg)>=.98:choices.append(dict(threshold=t,groups=len(gg),correct=len(cc),training_oof_precision=len(cc)/len(gg)))
    return choices[0] if choices else dict(threshold=None,groups=0)
def main():
    z=np.load(OUT/'trailer_references.npz');raw=z['vectors'];labels=z['labels'];groups=z['groups'];keys=z['keys'];x=balance(raw);s=x@x.T;m=[]
    for i in range(len(x)):
        a=groups!=groups[i];m.append(float(s[i,a&(labels=='attachment')].max()-s[i,a&(labels=='machine')].max()))
    m=np.array(m);nc=choose(m,labels,groups,.03);model=json.loads((OUT/'linear_region_candidate.json').read_bytes());metrics=json.loads((OUT/'linear_training.json').read_bytes());records=metrics['metrics']['oof_records'];assert len(records)==len(keys)
    probabilities=np.array([r['probabilities'] for r in records]);conditional=np.where((m>=.03)&(probabilities.argmax(axis=1)==0),probabilities[:,0],-np.inf);lc=choose(conditional,labels,groups,.95)
    path=OUT/'balanced_trailer_references.npz';np.savez_compressed(path,vectors=x.astype('float32'),labels=labels,groups=groups,keys=keys)
    config=model['trailer_reference'];config.update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),normalization='per_encoder_l2',threshold=nc['threshold'],oof_groups=nc['groups'],oof_correct_groups=nc.get('correct',0),supplement_minimum_margin=.03)
    config['supplement_calibration']['attachment']=lc
    def check(v):
        q=balance(v);ss=x@q;mm=float(ss[labels=='attachment'].max()-ss[labels=='machine'].max());v=v/max(np.linalg.norm(v),1e-9);lab,p,_=vote(config['supplement_head'],v*20);veto=nc['threshold'] is not None and mm>=nc['threshold'] or mm>=.03 and lab=='attachment' and qualified(config['supplement_calibration'],lab,p);return dict(margin=mm,label=lab,score=p,veto=bool(veto),uncertain=abs(mm)<.03 and not veto and not(lab=='machine' and qualified(config['supplement_calibration'],lab,p)))
    _,cv=vectors();data=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());main=[dict(key=r['key'],**check(cv[r['key']])) for split in ('test','winter_test','fresh_test','new_photos_test','round5_test') for r in data[split]]
    z=np.load(ROOT/'output/trailer_round5/trailer_features.npz');regions={str(k):check(v) for k,v in zip(z['keys'],z['values'],strict=True) if str(k).startswith(('region-control/','trailer5/'))}
    for n,v in [('fusion_region_candidate.json',model),('fusion_calibration.json',dict(reference=nc,conditional_linear=lc,explanation='Worst accepted region within every source group; mixed-label groups included. Linear decision additionally requires balanced OOF margin >= .03.')),('fusion_main_control.json',main),('fusion_region_control.json',regions)]:
        (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
    print('cal',nc,lc,'main veto',[(r['key'],r['margin']) for r in main if r['veto']],flush=True)
    print('critical',[(k,v) for k,v in regions.items() if k.startswith('region-control/')],flush=True)
if __name__=='__main__':main()
