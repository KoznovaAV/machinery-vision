import json,numpy as np
from pathlib import Path
from scripts.train_trailer_round5_verifier import vectors
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6';z=np.load(OUT/'trailer_references.npz');original=z['vectors'];labels=z['labels'];groups=z['groups'];keys=z['keys'];print('block norms',np.median(np.linalg.norm(original[:,:384],axis=1)),np.median(np.linalg.norm(original[:,384:],axis=1)),flush=True)
control=np.load(ROOT/'output/trailer_round5/trailer_features.npz');vv=dict(zip(map(str,control['keys']),control['values']));summaries={}
def norm(x,mode):
    if mode=='dino':x=x[...,:384]
    elif mode=='convnext':x=x[...,384:]
    elif mode=='balanced':x=np.concatenate([x[...,:384]/np.maximum(np.linalg.norm(x[...,:384],axis=-1,keepdims=True),1e-9),x[...,384:]/np.maximum(np.linalg.norm(x[...,384:],axis=-1,keepdims=True),1e-9)],axis=-1)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
for mode in ('raw','balanced','dino','convnext'):
    x=norm(original.copy(),mode);sim=x@x.T;m=[]
    for i in range(len(x)):
        a=groups!=groups[i];m.append(float(sim[i,a&(labels=='attachment')].max()-sim[i,a&(labels=='machine')].max()))
    m=np.array(m);choices=[]
    for t in sorted({.03}|{float(v) for v in m if v>=.03}):
        a=m>=t;ss=set(groups[a]);cc={g for g in ss if all(labels[i]=='attachment' for i in np.flatnonzero(a&(groups==g)))}
        if len(ss)>=5 and len(cc)/len(ss)>=.98:choices.append((t,len(ss),len(cc)))
    t,ss,cc=choices[0] if choices else (None,0,0);rr={}
    for k,v in vv.items():
        s=x@norm(v,mode);q=float(s[labels=='attachment'].max()-s[labels=='machine'].max());rr[k]=dict(margin=q,veto=t is not None and q>=t)
    summaries[mode]=dict(threshold=t,groups=ss,correct_groups=cc,controls=rr)
    print(mode,t,ss,'tank',rr['region-control/round5/r5b-103/0'],'red',rr['region-control/weather_expansion/resumable-photo-627531/0'],'combine',rr['region-control/model_wave2/tractorbg-293984/0'],flush=True)
(OUT/'fusion_experiment.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
