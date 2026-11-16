import json,numpy as np
from pathlib import Path
from scripts.train_trailer_round5_verifier import vectors
from src.models.mvp_inference import verify_trailer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5';data=json.loads((OUT/'dataset_manifest.json').read_bytes());b=json.loads((OUT/'reference_region_candidate.json').read_bytes());_,vv=vectors();rr=[]
for split in ('test','winter_test','fresh_test','new_photos_test','round5_test'):
    for r in data[split]:
        veto,evidence=verify_trailer(vv[r['key']],bundle=b);rr.append(dict(key=r['key'],split=split,veto=veto,evidence=evidence))
(OUT/'reference_main_control.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2),encoding='utf8');print('main machine veto proposals',[(r['key'],r['evidence'].get('margin')) for r in rr if r['veto']],flush=True)
