"""Manually reviewed collection; unclear weather and identities stay unlabelled."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from scripts.audit_round4_candidates import fingerprint,distance,canonical_source
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/robustness_round3'

def main():
    data=json.loads((ROOT/'output/winter_round2/dataset_manifest.json').read_bytes())
    old_rows=[r for name in ('train','family_only_train','test','winter_test','fresh_test','condition_rows') for r in data[name]]
    known={r['sha256']:fingerprint(ROOT/r['path']) for r in old_rows}
    protected_sources={canonical_source(r['source']) for r in old_rows if r.get('source','').startswith('http')}
    decisions={}
    def add(ids,model,split,condition,note,group=None):
        for key in ids.split():decisions[key]=dict(decision='accepted',model=model,split=split,condition=dict(zip(('dirty','snow','rain','wet'),condition)),note=note,group_override=group)
    add('nh005','other','train',(1,0,None,1),'New Holland of another series; large bonnet differs from T5000. Real deeply muddy machine. Exact model withheld.')
    add('nh010','other','train',(1,0,0,None),'Real Merlo telehandler; source and visible MERLO marking agree. Not a T5000.')
    add('nh012 nh020 nh022','nh_t5000','train',(0,0,0,0),'Dealer identifies T5040; matching real machine, cab and hood. Three views are one vehicle.','dealer-verschueren-t5040-1977h')
    add('nh013',None,'train',(1,0,None,None),'Real blue tractor with muddy tyres. Generic dealer category does not establish exact series.')
    add('nh014',None,'test',(None,0,0,None),'Source identifies T5040, but several machines overlap. Keep for scene/control only, no single-object model target.')
    add('nh016 nh021','nh_t5000','test',(0,0,0,None),'Readable T5040, registration 51-GS-67. Both listings show the same tractor.','nh-t5040-51gs67')
    add('nh018','other','train',(None,0,None,None),'Readable TD5040, not T5040. Listing title TD90 is inconsistent, exact model withheld.')
    add('nh019','nh_t5000','test',(0,0,0,None),'Real T5000-style hood/cab; manual-sale image is not evidence of exact variant. Series label only.')
    add('nh023','other','train',(1,0,1,1),'Readable TD5.90, not T5050. Mud adheres to tyres; water droplets and rain streaks visible. Keep series negative.')
    add('nh025','nh_t5000','train',(1,0,None,1),'Readable T5050 agrees with source. Dirt on tyres, wet windows and bonnet. Active falling rain not independently certain.')
    add('nh029','nh_t5000','train',(0,0,0,None),'Judicial auction identifies T5050; real complete vehicle, no substantial visible mud. Exact answer still withheld.')
    add('nh030','other','train',(1,0,0,None),'Source identifies T6050; visible model and proportions differ from T5000. Mud on tyres.')
    add('nh035','nh_t5000','test',(1,0,0,None),'Source identifies T5060; matching hood/cab. Clear adhering mud on front and rear tyres. Unseen vehicle for evaluation.')
    add('nh040 nh041','nh_t5000','test',(1,0,None,None),'Source identifies T5060. Same registration/vehicle across dusk views. Muddy tyres visible. Independent low-light control.','nh-t5060-paredes-dusk')
    add('nh043','nh_t5000','train',(0,0,None,None),'Dealer identifies T5060; hood/cab match. Clean body and tyres, wet pavement does not prove rain on machine.')
    add('curtis00',None,'test',(None,0,None,1),'Dealer T5050 interior with droplets on glass. Detail-only photo: wet-surface control, no model-training label.')
    add('nh048','nh_t5000','train',(0,0,0,None),'Readable T5050 and dealer title agree. Real two-wheel-drive configuration, no substantial visible mud.')
    add('nh053','nh_t5000','train',(1,0,None,None),'Dealer T5050; matching hood/cab, visible adhering mud on front and rear tyre treads. New source vehicle.')
    add('nh054','nh_t5000','test',(0,0,None,None),'Dealer T5050, matching real vehicle. Damp pavement alone is not a rain label. Held out by source.')
    add('nh057','nh_t5000','test',(None,0,0,None),'Real T5000-series hood/cab; T5060 dealer category alone does not certify exact variant. Light soil/dust on tyres is not labelled as heavy mud. Independent source held out.')
    decisions['nh016'].update(decision='rejected',note='Same registration 51-GS-67 as protected round5/r5b-20; different listing and view do not make an independent vehicle.')
    queue=json.loads((OUT/'queue.json').read_bytes())
    audit=[];accepted=[]
    for row in queue:
        path=OUT/'photos'/(row['id']+'.jpg')
        if not path.exists():continue
        try:
            with Image.open(path) as im:im.load()
        except OSError:continue
        dec=decisions.get(row['id'],dict(decision='rejected',note='No verified manual acceptance.'))
        record=dict(row,**dec,key='robust3/'+row['id'],path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        record['group']=dec.get('group_override') or 'source-'+hashlib.sha256(canonical_source(row['source']).encode()).hexdigest()[:16]
        fp=fingerprint(path)
        near=min((distance(fp,v),sha) for sha,v in known.items())
        if dec['decision']=='accepted':
            if record['sha256'] in known or near[0]<=6 or canonical_source(row['source']) in protected_sources:
                record.update(decision='duplicate_or_protected',nearest_sha256=near[1],protected_source=canonical_source(row['source']) in protected_sources)
            else:
                accepted.append(record);known[record['sha256']]=fp
        audit.append(record)
    data['new_review']=audit
    data['new_photos_test']=[dict(r,expected_family='nh_t5000_family' if r['model']=='nh_t5000' else 'unknown') for r in accepted if r['split']=='test' and r.get('model')]
    data['train'] += [r for r in accepted if r['split']=='train' and r.get('model')]
    data['condition_rows'] += [dict(r,role=r['split']) for r in accepted]
    data['family_training_policy']='Keep original known/unknown data; add only verified known-family examples to the shared family head. Unknown additions are reserved for a separate refusal experiment.'
    training=data['train']+data['family_only_train'];held=data['test']+data['winter_test']+data['fresh_test']+data['new_photos_test']
    for field in ('sha256','group'):
        assert not {r[field] for r in training} & {r[field] for r in held},field
    for name,obj in (('visual_decisions.json',decisions),('dataset_manifest.json',data)):
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('accepted',len(accepted),'NH T5000',sum(r.get('model')=='nh_t5000' for r in accepted),'new control',len(data['new_photos_test']),flush=True)
    print('accepted ids',[r['id'] for r in accepted],flush=True)

if __name__=='__main__':main()
