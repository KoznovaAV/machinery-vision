"""Audit changes and package only a non-regressing condition head."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from scripts.train_winter_round2 import ROOT,OUT,save

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    comparison=json.loads((OUT/'recognition_comparison.json').read_bytes())
    state=json.loads((OUT/'condition_comparison.json').read_bytes())
    gates=json.loads((OUT/'deployment_gates.json').read_bytes())
    rows={r['key']:r for k in ('train','family_only_train','test','winter_test','fresh_test','condition_rows') for r in data[k]}
    changes={}
    for split in ('test','winter_test','fresh_test'):
        old={r['key']:r for r in comparison['baseline'][split]['rows']}
        changes[split]=[dict(key=r['key'],path=rows[r['key']]['path'],before=old[r['key']],after=r)
            for r in comparison['candidate'][split]['rows'] if r['candidate']!=old[r['key']]['candidate'] or r['label']!=old[r['key']]['label']]
    save('changed_predictions.json',changes)
    failures=[r for split in ('winter_test','fresh_test') for r in comparison['candidate'][split]['rows'] if r['candidate']!=r['expected_family']]
    canvas=Image.new('RGB',(960,260*((len(failures)+2)//3)),color='white')
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(failures):
        x=(i%3)*320; y=(i//3)*260
        with Image.open(ROOT/rows[r['key']]['path']) as handle:
            photo=ImageOps.contain(ImageOps.exif_transpose(handle).convert('RGB'),(312,205))
        canvas.paste(photo,(x,y))
        draw.text((x+2,y+206),r['key'],fill='black')
        draw.text((x+2,y+222),'truth '+r['expected_family'],fill='black')
        draw.text((x+2,y+238),'pred '+r['candidate'],fill='black')
    canvas.save(OUT/'remaining_errors.jpg',quality=92)
    # Old and new source-group controls stay outside training for each task.
    training=data['train']+data['family_only_train']
    held=data['test']+data['winter_test']+data['fresh_test']
    for field in ('group','sha256'):
        assert not {r[field] for r in training} & {r[field] for r in held}
    for r in rows.values():
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['key']
    dest=ROOT/'artifacts/mvp'
    protected=('hierarchy_bundle.json','region_bundle.json')
    for filename in protected+('condition_bundle.json',):
        assert (dest/filename).read_bytes()==(OUT/'baseline'/filename).read_bytes(), 'Active artifact changed since frozen baseline: '+filename
    assert gates['condition_positive_evidence']
    candidate=json.loads((OUT/'condition_positive_evidence.json').read_bytes())
    assert candidate['calibration']['dirty']['0']['threshold'] is None
    # All other attributes remain byte-equivalent at the JSON-value level.
    old=json.loads((OUT/'baseline/condition_bundle.json').read_bytes())
    for key in ('heads','calibration','feature_sources'):
        for a,value in old.get(key,{}).items():
            if a!='dirty':
                assert candidate[key][a]==value
    target=dest/'condition_bundle.json'
    temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(target)
    (dest/'condition_metrics.json').write_text(json.dumps(dict(
        **state['positive_evidence'],evaluation='Expanded grouped control, same runtime decisions',
        manifest='output/winter_round2/dataset_manifest.json',
        note='Prototype release gate; reused controls and policy diagnostics are not blinded validation.'),ensure_ascii=False,indent=2),encoding='utf-8')
    release=dict(dataset_sha256=hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest(),
        recognition_promoted=False,condition_promoted=True,
        retained_recognition_reason='Both trained recognition candidates failed old-control non-regression gates.',
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.json')},
        new_accepted_photos=sum(r['decision']=='accepted' for r in data['new_review']))
    save('release_manifest.json',release)
    print('Released condition head only; recognition and region verifier retained.',flush=True)
    print('Changed original examples',len(changes['test']),flush=True)
    print('Family regressions',[(r['key'],r['before']['expected_family'],r['before']['candidate'],r['after']['candidate']) for r in changes['test'] if r['before']['candidate']==r['before']['expected_family'] and r['after']['candidate']!=r['after']['expected_family']],flush=True)
    print('Data integrity verified',len(rows),flush=True)

if __name__=='__main__':
    main()
