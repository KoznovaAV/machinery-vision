"""Train a discriminative supplement on the reviewed instance-reference bank."""
import json,numpy as np
from pathlib import Path
from scripts.train_mvp import train
from scripts.train_machinery_hierarchy import calibrate
from scripts.train_trailer_round5_verifier import vectors
from src.models.hierarchy import vote,qualified
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def main():
    z=np.load(OUT/'trailer_references.npz');x=z['vectors']*20;rows=[dict(key=str(k),group=str(g),model=str(l)) for k,g,l in zip(z['keys'],z['groups'],z['labels'],strict=True)];vv={r['key']:v for r,v in zip(rows,x,strict=True)}
    head,metrics=train(rows,vv,tune=False,min_groups=5);cal=calibrate(metrics['oof_records'],head['classes'],min_groups=5,target_precision=.98)
    for v in cal.values():
        if v['threshold'] is not None:v['threshold']=max(.95,v['threshold'])
    b=json.loads((OUT/'region_candidate.json').read_bytes());b['trailer_reference']['supplement_head']=head;b['trailer_reference']['supplement_calibration']=cal;b['trailer_reference']['supplement_scale']=20;b['trailer_reference']['supplement_minimum_margin']=0.
    old=np.load(OUT/'trailer_references.npz');ref=old['vectors'];labels=old['labels']
    def check(v):
        v=v/max(np.linalg.norm(v),1e-9);s=ref@v;margin=float(s[labels=='attachment'].max()-s[labels=='machine'].max());label,score,_=vote(head,v*20);return dict(margin=margin,label=label,score=score,veto=margin>=b['trailer_reference']['threshold'] or (margin>=0 and label=='attachment' and qualified(cal,label,score)))
    z=np.load(ROOT/'output/trailer_round5/trailer_features.npz');rr={str(k):check(v) for k,v in zip(z['keys'],z['values'],strict=True) if str(k).startswith(('region-control/','trailer5/'))}
    _,cv=vectors();data=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());main=[]
    for split in ('test','winter_test','fresh_test','new_photos_test','round5_test'):
        for r in data[split]:main.append(dict(key=r['key'],**check(cv[r['key']])))
    for n,v in [('linear_region_candidate.json',b),('linear_training.json',dict(metrics=metrics,calibration=cal)),('linear_control.json',rr),('linear_main_control.json',main)]:
        (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
    print('cal',cal,'C',metrics['c'],flush=True);print('main veto',[(r['key'],r['margin']) for r in main if r['veto']],flush=True);print('regions',[(k,round(v['margin'],3),v['label'],round(v['score'],4),v['veto']) for k,v in rr.items()],flush=True)
if __name__=='__main__':main()
