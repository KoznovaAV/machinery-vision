"""Group-validated reference matching for visually reviewed trailer feedback."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.train_trailer_round5_verifier import vectors
from src.models.hierarchy import vote
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def main():
    training=json.loads((OUT/'trailer_head_training.json').read_bytes());rows=training['rows'];_,vv=vectors();z=np.load(OUT/'trailer_features.npz');vv.update(zip(map(str,z['keys']),z['values'],strict=True))
    x=np.stack([vv[r['key']] for r in rows]);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9);labels=np.array([r['model'] for r in rows]);groups=np.array([r['group'] for r in rows])
    similarity=x@x.T; margins=[]
    for i,r in enumerate(rows):
        allowed=groups!=r['group'];neg=np.max(similarity[i,allowed&(labels=='attachment')]);pos=np.max(similarity[i,allowed&(labels=='machine')]);margins.append(float(neg-pos))
    margins=np.array(margins);choices=[]
    for t in sorted({.03}|{float(v) for v in margins if v>=.03}):
        accepted=margins>=t;selected=set(groups[accepted]);correct={g for g in selected if all(labels[i]=='attachment' for i in np.flatnonzero(accepted&(groups==g)))}
        if len(selected)>=5 and len(correct)/len(selected)>=.98:choices.append((t,len(selected),len(correct)))
    threshold,support,correct=choices[0] if choices else (None,0,0)
    dest=OUT/'trailer_references.npz';np.savez_compressed(dest,vectors=x.astype('float32'),labels=labels,groups=groups,keys=np.array([r['key'] for r in rows]))
    model=json.loads((OUT/'baseline/region_bundle.json').read_bytes());model['trailer_reference']=dict(path=dest.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),threshold=threshold,score='nearest_attachment_cosine_minus_nearest_machine_cosine',oof_groups=support,oof_correct_groups=correct,target_precision=.98,minimum_margin=.03,training_examples=len(rows),training_groups=len(set(groups)))
    z=np.load(ROOT/'output/robustness_round3/partial_features.npz');vv.update(zip(map(str,z['keys']),z['values'],strict=True))
    test=json.loads((OUT/'trailer_head_comparison.json').read_bytes());reports=[]
    for r in test:
        v=vv[r['key']];v=v/max(np.linalg.norm(v),1e-9);sim=x@v;neg=float(sim[labels=='attachment'].max());pos=float(sim[labels=='machine'].max());margin=neg-pos
        reports.append(dict(r,reference_margin=margin,reference_veto=threshold is not None and margin>=threshold))
    for n,value in [('reference_region_candidate.json',model),('reference_comparison.json',reports),('reference_oof.json',[dict(key=r['key'],group=r['group'],expected=r['model'],margin=float(m)) for r,m in zip(rows,margins,strict=True)])]:
        (OUT/n).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print('Reference threshold',threshold,'support',support,correct,flush=True)
    print('Machine fragments rejected',[(r['key'],r['reference_margin']) for r in reports if r.get('model')=='machine' and r['reference_veto']],flush=True)
    print('Controls',[(r.get('source_key',r['key']),round(r['reference_margin'],4),r['reference_veto']) for r in reports if r['key'].startswith(('region-control/','trailer5/'))],flush=True)
if __name__=='__main__':main()
