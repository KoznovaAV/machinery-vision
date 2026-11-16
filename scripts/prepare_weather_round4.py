"""Review weather candidates with identity and split protection by original machine."""
import copy,hashlib,json
from pathlib import Path
from PIL import Image
from scripts.audit_round4_candidates import canonical_source,fingerprint,distance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'

def main():
    data=copy.deepcopy(json.loads((ROOT/'output/robustness_round3/dataset_manifest.json').read_bytes()))
    old=[r for s in ('train','family_only_train','test','winter_test','fresh_test','new_photos_test','condition_rows') for r in data[s]]
    known={}
    for r in old:
        if r['sha256'] not in known:known[r['sha256']]=fingerprint(ROOT/r['path'])
    held=data['test']+data['winter_test']+data['fresh_test']+data['new_photos_test']+[r for r in data['condition_rows'] if r['role']=='test']
    held_sources={canonical_source(r['source']) for r in held if r.get('source','').startswith('http')}
    old_groups={canonical_source(r['source']):r['group'] for r in old if r.get('source','').startswith('http')}
    decisions={
        'w4_000':dict(decision='rejected',note='Exact duplicate of round5/r5b-0. Wet appearance is real but not new.'),
        'w4_001':dict(decision='rejected',note='Exact duplicate of round5/r5b-3.'),
        'w4_002':dict(decision='rejected',note='Exact duplicate of round5/r5b-2.'),
        'w4_008':dict(decision='accepted',model='other',split='test',condition=dict(dirty=0,snow=0,rain=0,wet=None),note='Readable TD5030 on bonnet, despite T5030 listing title. Different series from T5000; held-out source.'),
        'w4_012':dict(decision='accepted',model='other',split='train',condition=dict(dirty=1,snow=0,rain=None,wet=None),note='TD5030 body and source agree. Soil in tyre treads, muddy forecourt. Neither active rain nor wet tractor surface established.'),
        'w4_017':dict(decision='accepted',model='nh_t5000',split='train',condition=dict(dirty=None,snow=0,rain=0,wet=None),note='Readable T5070 and primary dealer description agree. Real orchard configuration with low cab. Light soil/dust does not establish substantial mud; no rain.'),
        'w4_021':dict(decision='accepted',model='other',split='test',condition=dict(dirty=1,snow=0,rain=1,wet=1),note='Modern NH hood/cab differs from T5000; exact variant withheld. Soil on wheels/steps, water on glass and bonnet, rain streaks across wooden wall and tractor. New source held out.'),
    }
    audit=[];accepted=[]
    for row in json.loads((OUT/'queue.json').read_bytes()):
        path=OUT/'photos'/(row['id']+'.jpg')
        if not path.exists():continue
        try:
            with Image.open(path) as im:im.load()
        except OSError:continue
        decision=decisions.get(row['id'],dict(decision='rejected',note='No verified visual decision.'))
        item=dict(row,**decision,key='weather4/'+row['id'],path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        source=canonical_source(row['source']);item['group']=old_groups.get(source,'source-'+hashlib.sha256(source.encode()).hexdigest()[:16])
        fp=fingerprint(path);near=min((distance(fp,v),sha) for sha,v in known.items())
        if item['decision']=='accepted':
            if item['sha256'] in known or near[0]<=6 or source in held_sources:
                item.update(decision='duplicate_or_protected',nearest_sha256=near[1])
            else:accepted.append(item);known[item['sha256']]=fp
        audit.append(item)
    data['new_review']=audit
    data['round4_test']=[dict(r,expected_family='nh_t5000_family' if r['model']=='nh_t5000' else 'unknown') for r in accepted if r['split']=='test']
    data['train'] += [r for r in accepted if r['split']=='train']
    data['condition_rows'] += [dict(r,role=r['split']) for r in accepted]
    train=data['train']+data['family_only_train'];test=data['test']+data['winter_test']+data['fresh_test']+data['new_photos_test']+data['round4_test']
    legacy_condition_groups={r['group'] for r in train}&{r['group'] for r in held if r.get('role')=='test'}
    data['legacy_condition_recognition_overlap']=sorted(legacy_condition_groups)
    for field in ('group','sha256'):
        assert not {r[field] for r in accepted if r['split']=='train'} & {r[field] for r in held},field
    for field in ('group','sha256'):assert not {r[field] for r in train} & {r[field] for r in test},field
    for name,obj in (('visual_decisions.json',decisions),('dataset_manifest.json',data)):
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Accepted',len(accepted),[r['id'] for r in accepted],'T5000',sum(r['model']=='nh_t5000' for r in accepted),flush=True)

if __name__=='__main__':main()
