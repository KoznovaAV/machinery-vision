import json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from scripts.predict_machinery_hierarchy import detector_model
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
torch.set_num_threads(5);processor,model=detector_model();audit=json.loads((OUT/'region_control_audit.json').read_bytes());cases={k:ROOT/r['path'] for k,r in audit.items() if r['ignored']};cases['wet']=ROOT/'output/weather_round4/photos/w4_021.jpg';rr={}
for key,path in cases.items():
    with Image.open(path) as h:im=ImageOps.exif_transpose(h).convert('RGB')
    x=processor(images=im,text='tractor cabin. combine harvester cabin.',size={'shortest_edge':320,'longest_edge':544},return_tensors='pt')
    with torch.inference_mode():p=model(**x)
    r=processor.post_process_grounded_object_detection(p,x.input_ids,threshold=.20,text_threshold=.2,target_sizes=[(im.height,im.width)])[0]
    rr[key]=[dict(box=b,score=s,label=str(l)) for b,s,l in zip(r['boxes'].tolist(),r['scores'].tolist(),r.get('text_labels',r.get('labels',[])),strict=True)]
    (OUT/'cab_diagnosis.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2),encoding='utf8');print(key,rr[key][:5],flush=True)
