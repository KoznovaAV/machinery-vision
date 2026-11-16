"""Train an edge-fragment verifier from grouped crops of reviewed real machines."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageDraw
from scripts.train_winter_round2 import vectors
from scripts.build_winter_features import crop_path,fused
from scripts.train_mvp import train
from src.models.hierarchy import vote,qualified
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/robustness_round3'

def main():
    torch.set_num_threads(5)
    data=json.loads((ROOT/'output/winter_round2/dataset_manifest.json').read_bytes())
    old=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes())
    full,crop=vectors()
    selected=defaultdict(list)
    for r in sorted(data['train'],key=lambda x:x['sha256']):
        if r['model']=='other':continue
        if len(selected[r['model']])<3 and r['group'] not in {v['group'] for v in selected[r['model']]}:
            selected[r['model']].append(r)
    rows=[]; files={}
    for r in [x for rr in selected.values() for x in rr]+old['region_negatives']:
        label='attachment' if r in old['region_negatives'] else 'machine'
        with Image.open(ROOT/r['path'] if label=='attachment' else crop_path(r)) as handle:
            image=ImageOps.exif_transpose(handle).convert('RGB')
        for side in ('left35','right35','left70','right70'):
            fraction=.35 if side.endswith('35') else .70
            box=(0,0,int(image.width*fraction),image.height) if side.startswith('left') else (int(image.width*(1-fraction)),0,image.width,image.height)
            path=OUT/'partial_views'/r['sha256']/(side+'.jpg')
            path.parent.mkdir(parents=True,exist_ok=True)
            image.crop(box).save(path,quality=95)
            key='partial/'+r['key']+'/'+side
            rows.append(dict(key=key,group=r['group'],model=label,path=path.relative_to(ROOT).as_posix(),source_key=r['key'],role='train'))
            files[key]=path
    controls=[]
    live=json.loads((ROOT/'output/winter_round2/live_web_verification.json').read_bytes())
    for case in ('multiple','attachment'):
        item=next(r for r in live if r['case']==case)
        pred=json.loads(Path(item['result']).read_bytes())
        region=pred['ignored_regions'][0]
        path=OUT/'partial_control'/(case+'.jpg');path.parent.mkdir(exist_ok=True)
        with Image.open(item['source']) as handle:
            image=ImageOps.exif_transpose(handle).convert('RGB')
        image.crop(region['bbox']).save(path,quality=95)
        row=dict(key='control/'+case,group='user-'+case,model='machine' if case=='multiple' else 'attachment',path=path.relative_to(ROOT).as_posix(),role='test')
        row.update(parent_sha256=hashlib.sha256(Path(item['source']).read_bytes()).hexdigest(),bbox=region['bbox'])
        if case=='multiple':
            row['role']='feedback_train'
            row['manual_note']='Visible blue engine hood, chassis and wheel belong to the cropped third real tractor. Feedback case, never independent validation.'
        controls.append(row); files[row['key']]=path
    selected_test={}
    for r in sorted(data['test'],key=lambda r:r['sha256']):
        if r['model']!='other':selected_test.setdefault(r['model'],r)
    for r in selected_test.values():
        with Image.open(crop_path(r)) as handle:
            image=ImageOps.exif_transpose(handle).convert('RGB')
        for side in ('left','right'):
            box=(0,0,int(image.width*.35),image.height) if side=='left' else (int(image.width*.65),0,image.width,image.height)
            path=OUT/'partial_control'/(r['sha256']+'-'+side+'.jpg')
            image.crop(box).save(path,quality=95)
            row=dict(key='control/'+r['key']+'/'+side,group=r['group'],model='machine',path=path.relative_to(ROOT).as_posix(),role='test')
            controls.append(row);files[row['key']]=path
    for offset in range(0,len(rows),32):
        batch=rows[offset:offset+32]; sheet=Image.new('RGB',(1000,190*((len(batch)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(batch):
            thumb=ImageOps.contain(Image.open(ROOT/r['path']),(244,155));x=i%4*250;y=i//4*190;sheet.paste(thumb,(x,y));draw.text((x+2,y+157),str(offset+i)+' '+r['model'],fill='black')
        sheet.save(OUT/('partial_review_'+str(offset)+'.jpg'),quality=90)
    cache=OUT/'partial_features.npz'
    signature=hashlib.sha256('|'.join(k+':'+hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sorted(files.items())).encode()).hexdigest()
    if cache.exists() and str(np.load(cache)['signature'])==signature:
        z=np.load(cache); extra=dict(zip(z['keys'],z['values'],strict=True))
    else:
        extra=fused(files);np.savez_compressed(cache,signature=signature,keys=list(extra),values=list(extra.values()))
    vv=dict(crop);vv.update(extra)
    # Original uncropped positive examples anchor the verifier. All recognition controls stay excluded.
    base=[dict(r,model='attachment') for r in old['region_negatives']]
    # This supplemental head learns fragments; the original verifier retains whole-machine responsibility.
    base += [dict(r,model='machine') for rr in selected.values() for r in rr]
    # Reviewed front/lower real-machine regions provide actual component appearance.
    training_by_key={r['key']:r for r in data['train'] if r['model']!='other'}
    part_rows=[]
    for filename,kind in (('front_features.npz','front'),('features.npz','lower')):
        z=np.load(ROOT/'output/tractor_parts'/filename)
        for key,value in zip(z['keys'],z['values'],strict=True):
            source=str(key) if kind=='front' else str(key).removesuffix('/lower')
            if kind=='lower' and not str(key).endswith('/lower'):continue
            if source not in training_by_key:continue
            original=training_by_key[source]
            new_key='physical-part/'+kind+'/'+source
            vv[new_key]=value
            part_rows.append(dict(key=new_key,source_key=source,group=original['group'],model='machine',role='train',feature_cache='output/tractor_parts/'+filename))
    base += part_rows
    feedback=[r for r in controls if r['role']=='feedback_train']
    protected_hashes={r['sha256'] for s in ('test','winter_test','fresh_test') for r in data[s]}
    assert not {r['parent_sha256'] for r in feedback} & protected_hashes
    base += feedback
    head,summary=train(base+rows,vv,min_groups=5)
    threshold=summary['thresholds']['machine']['threshold']
    if threshold is not None:summary['thresholds']['machine']['threshold']=max(.95,threshold)
    reports=[]
    for r in controls:
        label,score,_=vote(head,vv[r['key']])
        reports.append(dict(**r,label=label,score=score,accepted=label=='machine' and qualified(summary['thresholds'],'machine',score)))
    old_bundle=json.loads((OUT/'baseline/region_bundle.json').read_bytes())
    bundle=dict(old_bundle,partial_head=head,partial_calibration=summary['thresholds'],partial_training=summary,
        partial_policy='edge_only_manual_review',partial_minimum_score=.95)
    for name,obj in (('partial_region_candidate.json',bundle),('partial_training_manifest.json',dict(train=base+rows,test=[r for r in controls if r['role']=='test'],feedback_regression=[r for r in controls if r['role']=='feedback_train'])),('partial_control_metrics.json',reports)):
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('partial',reports[:2],'held positives',sum(r['model']=='machine' and r['accepted'] for r in reports),'/',sum(r['model']=='machine' for r in reports),'OOF',summary['grouped_oof_balanced_accuracy'],'threshold',summary['thresholds']['machine'],flush=True)

if __name__=='__main__':main()
