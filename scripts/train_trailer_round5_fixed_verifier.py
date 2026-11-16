"""Train an additional image verifier; old region/partial heads are immutable."""
import hashlib,json,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from scripts.build_winter_features import fused
from scripts.train_weather_round4_specialist import load_vectors
from scripts.train_mvp import train
from scripts.train_machinery_hierarchy import calibrate
from src.models.hierarchy import vote,qualified
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def vectors():
    full,crop=load_vectors()
    for p in ('output/weather_round4/features.npz','output/trailer_round5/features.npz'):
        z=np.load(ROOT/p)
        for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    return full,crop

def main():
    torch.set_num_threads(5);_,vv=vectors();data=json.loads((ROOT/'output/robustness_round3/partial_training_manifest.json').read_bytes())
    z=np.load(ROOT/'output/robustness_round3/partial_features.npz');vv.update(zip(map(str,z['keys']),z['values'],strict=True))
    for fn,kind in [('front_features.npz','front'),('features.npz','lower')]:
        z=np.load(ROOT/'output/tractor_parts'/fn)
        for k,v in zip(z['keys'],z['values'],strict=True):
            if kind=='lower' and not str(k).endswith('/lower'):continue
            source=str(k).removesuffix('/lower') if kind=='lower' else str(k)
            vv['physical-part/'+kind+'/'+source]=v
    new=json.loads((OUT/'new_region_training.json').read_bytes())
    augmented=[]
    for row in new:
        with Image.open(ROOT/row['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB')
        for side in ('left35','right35','left70','right70'):
            f=.35 if side.endswith('35') else .70
            box=(0,0,int(im.width*f),im.height) if side.startswith('left') else (int(im.width*(1-f)),0,im.width,im.height)
            dest=OUT/'region_train'/(row['key'].replace('/','_')+'_'+side+'.jpg');im.crop(box).save(dest,quality=95)
            augmented.append(dict(row,key=row['key']+'/'+side,path=dest.relative_to(ROOT).as_posix(),augmentation='real_photo_crop',sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    new+=augmented
    held_trailers=json.loads((OUT/'new_region_test.json').read_bytes())
    files={r['key']:ROOT/r['path'] for r in new+held_trailers}
    audit=json.loads((OUT/'region_control_audit.json').read_bytes());extra=[]
    for k,r in audit.items():
        for i,drop in enumerate(r['ignored']):
            key='region-control/'+k+'/'+str(i);dest=OUT/'region_control'/(hashlib.sha256(key.encode()).hexdigest()+'.jpg');dest.parent.mkdir(exist_ok=True)
            with Image.open(ROOT/r['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB').crop(drop['bbox'])
            im.save(dest,quality=95);files[key]=dest;extra.append(dict(key=key,source_key=k,path=dest.relative_to(ROOT).as_posix(),bbox=drop['bbox']))
    # This wet-trailer example remains test-only and is never used to fit the head.
    wet=json.loads((OUT/'semantic_diagnosis.json').read_bytes())['wet_trailer']
    dest=OUT/'region_control/wet_trailer.jpg'
    with Image.open(wet['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB').crop(wet['regions'][1]['box'])
    im.save(dest,quality=95);files['region-control/wet_trailer']=dest;extra.append(dict(key='region-control/wet_trailer',source_key='weather4/w4_021',path=dest.relative_to(ROOT).as_posix(),bbox=wet['regions'][1]['box']))
    cache=OUT/'trailer_features.npz';signature=hashlib.sha256('|'.join(k+hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sorted(files.items())).encode()).hexdigest()
    if cache.exists() and str(np.load(cache)['signature'])==signature:
        z=np.load(cache);features=dict(zip(map(str,z['keys']),z['values'],strict=True))
    else:
        features=fused(files);np.savez_compressed(cache,signature=signature,keys=list(features),values=list(features.values()))
    vv.update(features)
    manifest=json.loads((OUT/'dataset_manifest.json').read_bytes())
    original=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes())
    rows=[dict(r,model='machine') for r in manifest['train']]+[dict(r,model='attachment') for r in original['region_negatives']]+new
    missing=[r['key'] for r in rows if r['key'] not in vv];assert not missing,missing[:10]
    assert len({r['key'] for r in rows})==len(rows)
    protected={r['group'] for r in data['test']};assert not {r['group'] for r in rows}&protected
    head,metrics=train(rows,vv,tune=False,min_groups=5);cal=calibrate(metrics['oof_records'],head['classes'],min_groups=5,target_precision=.98)
    for v in cal.values():
        if v['threshold'] is not None:v['threshold']=max(.95,v['threshold'])
    old=json.loads((OUT/'baseline/region_bundle.json').read_bytes());candidate=dict(old,trailer_head=head,trailer_calibration=cal,trailer_minimum_score=.95,trailer_policy='semantic_and_image_agreement',trailer_training=metrics)
    reports=[]
    for r in data['test']+held_trailers+extra:
        label,score,_=vote(head,vv[r['key']]);reports.append(dict(r,label=label,score=score,veto=label=='attachment' and qualified(cal,label,score)))
    for n,v in [('region_candidate.json',candidate),('trailer_head_comparison.json',reports),('trailer_head_training.json',dict(rows=rows,calibration=cal,metrics=metrics))]:save(n,v)
    print('fresh trailer control',[(r['key'],r['label'],round(r['score'],4),r['veto']) for r in reports if r['key'].startswith('trailer5/')],flush=True)
    print('calibration',cal,flush=True)
    print('machine fragments rejected',[(r['key'],r['score']) for r in reports if r.get('model')=='machine' and r['veto']],flush=True)
    print('semantic candidates',[(r['source_key'],r['label'],round(r['score'],4),r['veto']) for r in reports if r['key'].startswith('region-control')],flush=True)
if __name__=='__main__':main()
