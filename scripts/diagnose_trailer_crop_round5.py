import json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from scripts.predict_machinery_hierarchy import detector_model,attachment_semantics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
torch.set_num_threads(5);det=detector_model();audit=json.loads((OUT/'region_control_audit.json').read_bytes());original=json.loads((OUT/'semantic_diagnosis.json').read_bytes());cases=[]
for key,row in audit.items():
    for n,v in enumerate(row['ignored']):cases.append((key+'/%d'%n,ROOT/row['path'],v['bbox']))
for n,v in enumerate(original['wet_trailer']['regions']):cases.append(('wet/%d'%n,Path(original['wet_trailer']['path']),v['box']))
result={}
for key,path,box in cases:
    with Image.open(path) as h:im=ImageOps.exif_transpose(h).convert('RGB').crop(box)
    rr=attachment_semantics(im,det);result[key]=dict(size=im.size,semantic=rr)
    print(key,rr,flush=True)
    (OUT/'crop_semantic_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
