from pathlib import Path
import json,hashlib
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
manifest=json.loads((OUT/'dataset_manifest.json').read_bytes());old=manifest.get('round4_test') or manifest.get('round4_feedback_regression');manifest['round4_feedback_regression']=old;manifest['round4_test']=[]
for r in manifest['condition_rows']:
    if r['key']=='weather4/w4_021':r['role']='feedback_regression'
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
regions=json.loads((OUT/'new_region_training.json').read_bytes())
previous_test=OUT/'new_region_test.json'
if previous_test.exists():regions=list({r['key']:r for r in regions+json.loads(previous_test.read_bytes())}.values())
regions=[r for r in regions if r['key']!='feedback/wet_trailer']
held=[dict(r,role='test') for r in regions if r['key'] in ('trailer5/neg_01','trailer5/neg_05')];regions=[r for r in regions if r['key'] not in {x['key'] for x in held}]
source=old[0];box=[1967,109,2560,1204]
with Image.open(ROOT/source['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB').crop(box)
dest=OUT/'region_train/feedback_wet_trailer.jpg';im.save(dest,quality=95)
regions.append(dict(key='feedback/wet_trailer',source=source['source'],group=source['group'],model='attachment',role='feedback_train',path=dest.relative_to(ROOT).as_posix(),parent_path=source['path'],parent_sha256=source['sha256'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bbox=box,manual_note='Only the trailer proposal, not a second tractor. Known diagnostic case deliberately moved from control to training feedback.'))
(OUT/'new_region_training.json').write_text(json.dumps(regions,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'new_region_test.json').write_text(json.dumps(held,ensure_ascii=False,indent=2),encoding='utf8')
p=ROOT/'scripts/train_trailer_round5_whole_verifier.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("files={r['key']:ROOT/r['path'] for r in new}","held_trailers=json.loads((OUT/'new_region_test.json').read_bytes())\n    files={r['key']:ROOT/r['path'] for r in new+held_trailers}")
s=s.replace("for r in data['test']+extra:","for r in data['test']+held_trailers+extra:")
s=s.replace("save(n,v)\n    print('calibration'", "save(n,v)\n    print('fresh trailer control',[(r['key'],r['label'],round(r['score'],4),r['veto']) for r in reports if r['key'].startswith('trailer5/')],flush=True)\n    print('calibration'")
p.write_text(s,encoding='utf8')
