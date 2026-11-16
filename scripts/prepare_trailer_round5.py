"""Manual review, duplicate protection and grouped data for trailer round 5."""
import copy,hashlib,json
from pathlib import Path
from PIL import Image,ImageOps
from scripts.audit_round4_candidates import fingerprint,distance,canonical_source
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def save(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    data=copy.deepcopy(json.loads((ROOT/'output/weather_round4/dataset_manifest.json').read_bytes()))
    old=[r for s in ('train','family_only_train','test','winter_test','fresh_test','new_photos_test','round4_test','condition_rows') for r in data[s]]
    fps={r['sha256']:fingerprint(ROOT/r['path']) for r in old};group_by_source={canonical_source(r['source']):r['group'] for r in old if r.get('source','').startswith('http')}
    held=[r for s in ('test','winter_test','fresh_test','new_photos_test','round4_test') for r in data[s]]
    protected={r['group'] for r in held}|{r['group'] for r in data['condition_rows'] if r['role']=='test'}
    decisions={
        't5_003':dict(decision='rejected',note='Exact duplicate of existing wet T5050; not new.'),
        't5_c2':dict(decision='rejected',note='Exact duplicate of round5/r5-10; not new.'),
        't5_009':dict(decision='accepted',model='nh_t5000',split='train',condition=dict(dirty=1,snow=0,rain=0,wet=None),note='Readable T5070; dry mud/soil coating on tyres, chassis and loader. Indoors, not a rain example.'),
        't5_b0':dict(decision='accepted',model='nh_t5000',split='train',condition=dict(dirty=1,snow=0,rain=0,wet=None),note='Same auction T5070 from front; visible soil coating. Same machine group, never independent control.'),
        't5_b1':dict(decision='accepted',model='nh_t5000',split='train',condition=dict(dirty=1,snow=0,rain=0,wet=None),note='Same real T5070 side view with loader; mud on wheel rims and chassis. Same source group.'),
        't5_b2':dict(decision='accepted',model='nh_t5000',split='train',condition=dict(dirty=1,snow=0,rain=0,wet=None),note='Same real T5070 rear/side view, soil on wheels and rear attachment. Same source group.'),
        't5_c3':dict(decision='accepted',model='nh_t5000',split='test',condition=dict(dirty=0,snow=0,rain=None,wet=None),note='Real T5060-series exterior with matching old-style T5000 hood and public listing. Surface moisture/active rain cannot be confirmed. New source held out.'),
    }
    accepted=[];review=[];negative_rows=[]
    for r in json.loads((OUT/'downloaded.json').read_bytes()):
        if r['status']!='downloaded':continue
        path=OUT/'photos'/(r['id']+'.jpg');fp=fingerprint(path);source=canonical_source(r['source']);group=group_by_source.get(source,'source-'+hashlib.sha256(source.encode()).hexdigest()[:16])
        if r['id'].startswith('neg_'):
            with Image.open(path) as h:im=ImageOps.exif_transpose(h).convert('RGB')
            ratios={'neg_02':(.18,0,1,1),'neg_04':(.43,.24,1,.88)};ratio=ratios.get(r['id'],(0,0,1,1));box=[int(v*dim) for v,dim in zip(ratio,(im.width,im.height,im.width,im.height))]
            region=im.crop(box);dest=OUT/'region_train'/(r['id']+'.jpg');dest.parent.mkdir(exist_ok=True);region.save(dest,quality=95)
            row=dict(r,key='trailer5/'+r['id'],path=dest.relative_to(ROOT).as_posix(),parent_path=path.relative_to(ROOT).as_posix(),parent_sha256=fp['sha256'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),group=group,model='attachment',role='train',bbox=box,manual_note='Real non-powered agricultural trailer; tractor excluded from training crop.')
            assert group not in protected;negative_rows.append(row);continue
        dec=decisions.get(r['id'],dict(decision='rejected',note='Not manually verified.'))
        item=dict(r,**dec,key='trailer5/'+r['id'],path=path.relative_to(ROOT).as_posix(),sha256=fp['sha256'],group=group)
        near=min((distance(fp,v),sha) for sha,v in fps.items())
        if item['decision']=='accepted' and (fp['sha256'] in fps or near[0]<=6 or (item['split']=='train' and group in protected)):
            item.update(decision='duplicate_or_protected',nearest_sha256=near[1],phash_distance=near[0])
        if item['decision']=='accepted':accepted.append(item);fps[fp['sha256']]=fp
        review.append(item)
    data['new_review']=review;data['round5_test']=[dict(r,expected_family='nh_t5000_family') for r in accepted if r['split']=='test']
    data['train'] += [r for r in accepted if r['split']=='train'];data['condition_rows'] += [dict(r,role=r['split']) for r in accepted]
    held+=data['round5_test'];training=data['train']+data['family_only_train']
    for field in ('group','sha256'):assert not {r[field] for r in training}&{r[field] for r in held}
    save('dataset_manifest.json',data);save('visual_decisions.json',decisions);save('new_region_training.json',negative_rows)
    print('accepted machinery',[(r['id'],r['split']) for r in accepted],'trailer source groups',len({r['group'] for r in negative_rows}),flush=True)
if __name__=='__main__':main()
