import json
from pathlib import Path
from src.models.mvp_inference import verify_trailer
import scripts.predict_machinery_hierarchy as p
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
b=json.loads((OUT/'reference_region_candidate.json').read_bytes());p.verify_trailer=lambda v:verify_trailer(v,bundle=b)
result=p.run(ROOT/'output/weather_round4/photos/w4_021.jpg',OUT/'baseline/hierarchy_bundle.json',OUT/'tight_feedback_check/prediction.json')
print('objects',len(result['objects']),'ignored',result['ignored_regions'],flush=True)
assert len(result['objects'])==1 and result['ignored_regions'][0]['reason']=='semantic_trailer'
