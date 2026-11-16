import json
from pathlib import Path
from scripts.audit_round4_candidates import fingerprint,distance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6';d=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());old={r['sha256']:r for split in ('train','family_only_train','test','winter_test','fresh_test','new_photos_test','condition_rows') for r in d[split]};fp={sha:fingerprint(ROOT/r['path']) for sha,r in old.items()};rr=[]
for row in json.loads((OUT/'manual_external_review.json').read_bytes())+[json.loads((OUT/'new_tanker_control.json').read_bytes())]:
    p=ROOT/row['path'];value=fingerprint(p);nearest=min((distance(value,v),sha) for sha,v in fp.items());rr.append(dict(key=row['key'],sha256=value['sha256'],nearest_distance=nearest[0],nearest_key=old[nearest[1]]['key'],exact_duplicate=value['sha256'] in old));assert value['sha256'] not in old and nearest[0]>6,row['key']
(OUT/'external_duplicate_audit.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2),encoding='utf8');print(rr,flush=True)
