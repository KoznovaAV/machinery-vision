import json,numpy as np
from pathlib import Path
import torch
from scripts.build_winter_features import fused
from src.models.mvp_inference import verify_trailer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6';b=json.loads((OUT/'fusion_region_candidate.json').read_bytes());z=np.load(ROOT/'output/robustness_round3/partial_features.npz');vv=dict(zip(map(str,z['keys']),z['values']));data=json.loads((ROOT/'output/robustness_round3/partial_training_manifest.json').read_bytes());rr=[]
for r in data['test']:
    veto,evidence=verify_trailer(vv[r['key']],bundle=b);rr.append(dict(key=r['key'],expected=r['model'],veto=veto,evidence=evidence))
torch.set_num_threads(5);control=json.loads((OUT/'new_tanker_control.json').read_bytes());value=fused({'new_tanker':ROOT/control['region_path']})['new_tanker'];veto,evidence=verify_trailer(value,bundle=b)
(OUT/'independent_region_checks.json').write_text(json.dumps(dict(partial_control=rr,new_tanker=dict(key=control['key'],veto=veto,evidence=evidence)),ensure_ascii=False,indent=2),encoding='utf8');print('machine fragments veto',[(r['key'],r['evidence']) for r in rr if r['expected']=='machine' and r['veto']],flush=True);print('new tanker',veto,evidence,flush=True)
