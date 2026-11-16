"""Expand reference matching with reviewed training-only trailers/tankers and machine fragments."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.build_winter_features import fused
from scripts.train_trailer_round5_verifier import vectors
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    torch.set_num_threads(5);base=OUT/'baseline/trailer_references.npz';old=np.load(base);keys=list(map(str,old['keys']));labels=list(map(str,old['labels']));groups=list(map(str,old['groups']));xx=list(old['vectors'])
    remove=keys.index('round6/r6s-r5-2')
    for arr in (keys,labels,groups,xx):arr.pop(remove)
    review=json.loads((OUT/'manual_region_review.json').read_bytes());rows=[dict(r,model=r['manual_label']) for r in review if r['decision']=='accepted'];rows+=json.loads((OUT/'manual_instance_annotations.json').read_bytes())
    files={r['key']:ROOT/r['path'] for r in rows}
    for r in json.loads((OUT/'manual_external_review.json').read_bytes()):
        with Image.open(ROOT/r['path']) as h:im=h.convert('RGB');f=r['crop_fraction'];im=im.crop(tuple(int(v*d) for v,d in zip(f,(im.width,im.height,im.width,im.height))))
        for name,fraction in [('whole',(0,0,1,1)),('left70',(0,0,.7,1)),('right70',(.3,0,1,1)),('left35',(0,0,.35,1)),('right35',(.65,0,1,1))]:
            key=r['key']+'/'+name;path=OUT/'region_train'/(r['id']+'_'+name+'.jpg');path.parent.mkdir(exist_ok=True);im.crop(tuple(int(v*d) for v,d in zip(fraction,(im.width,im.height,im.width,im.height)))).save(path,quality=95);row=dict(r,key=key,path=path.relative_to(ROOT).as_posix());rows.append(row);files[key]=path
    cache=OUT/'region_features.npz'
    if cache.exists() and set(map(str,np.load(cache)['keys']))==set(files):z=np.load(cache);extra=dict(zip(map(str,z['keys']),z['values'],strict=True))
    else:extra=fused(files);np.savez_compressed(cache,keys=list(extra),values=list(extra.values()))
    for r in rows:
        v=extra[r['key']];xx.append(v/max(np.linalg.norm(v),1e-9));keys.append(r['key']);labels.append(r['model']);groups.append(r['group'])
    x=np.stack(xx).astype('float32');labels=np.array(labels);groups=np.array(groups);sim=x@x.T;margins=[]
    for i in range(len(keys)):
        allowed=groups!=groups[i];margins.append(float(sim[i,allowed&(labels=='attachment')].max()-sim[i,allowed&(labels=='machine')].max()))
    margins=np.array(margins);choices=[]
    for t in sorted({.03}|{float(v) for v in margins if v>=.03}):
        accepted=margins>=t;selected=set(groups[accepted]);correct={g for g in selected if all(labels[i]=='attachment' for i in np.flatnonzero(accepted&(groups==g)))}
        if len(selected)>=5 and len(correct)/len(selected)>=.98:choices.append((t,len(selected),len(correct)))
    threshold,support,correct=choices[0] if choices else (None,0,0);path=OUT/'trailer_references.npz';np.savez_compressed(path,vectors=x,keys=np.array(keys),labels=labels,groups=groups)
    bundle=json.loads((OUT/'baseline/region_bundle.json').read_bytes());bundle['trailer_reference'].update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=threshold,oof_groups=support,oof_correct_groups=correct,training_examples=len(keys),training_groups=len(set(groups)))
    save('region_candidate.json',bundle);save('reference_training.json',dict(additions=rows,oof=[dict(key=k,group=g,label=l,margin=float(m)) for k,g,l,m in zip(keys,groups,labels,margins,strict=True)]))
    data=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());held=[r for s in ('test','winter_test','fresh_test','new_photos_test','round5_test') for r in data[s]];assert not set(groups)&{r['group'] for r in held}
    _,crop=vectors();rr=[]
    for r in held:
        v=crop[r['key']];v=v/max(np.linalg.norm(v),1e-9);s=x@v;m=float(s[labels=='attachment'].max()-s[labels=='machine'].max());rr.append(dict(key=r['key'],margin=m,veto=threshold is not None and m>=threshold))
    z=np.load(ROOT/'output/trailer_round5/trailer_features.npz');control=dict(zip(map(str,z['keys']),z['values'],strict=True))
    regions=[]
    for r in json.loads((ROOT/'output/trailer_round5/reference_comparison.json').read_bytes()):
        if r['key'] not in control:continue
        v=control[r['key']];s=x@(v/max(np.linalg.norm(v),1e-9));m=float(s[labels=='attachment'].max()-s[labels=='machine'].max());regions.append(dict(key=r['key'],source_key=r.get('source_key'),margin=m,veto=threshold is not None and m>=threshold))
    save('main_control.json',rr);save('region_comparison.json',regions);print('threshold',threshold,'OOF',support,correct,'main vetoes',[(r['key'],r['margin']) for r in rr if r['veto']],flush=True);print('regions',[(r['source_key'] or r['key'],round(r['margin'],3),r['veto']) for r in regions],flush=True)
if __name__=='__main__':main()
