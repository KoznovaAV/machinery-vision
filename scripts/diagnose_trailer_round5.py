"""Inspect independent semantic evidence before changing machine detection."""
import json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from scripts.predict_machinery_hierarchy import detector_model,detect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def semantic(image,detector):
    processor,model=detector
    inputs=processor(images=image,text='tractor. combine harvester. bulldozer. trailer. agricultural implement.',size={'shortest_edge':320,'longest_edge':544},return_tensors='pt')
    with torch.inference_mode():pred=model(**inputs)
    r=processor.post_process_grounded_object_detection(pred,inputs.input_ids,threshold=.15,text_threshold=.2,target_sizes=[(image.height,image.width)])[0]
    return [dict(box=b,score=s,label=str(l)) for b,s,l in zip(r['boxes'].tolist(),r['scores'].tolist(),r.get('text_labels',r.get('labels',[])),strict=True)]
def main():
    torch.set_num_threads(5);d=detector_model(); rows={}
    data=json.loads((ROOT/'output/mvp/dataset_manifest.json').read_bytes())
    cases={r['key']:ROOT/r['path'] for r in data['region_negatives']}
    cases.update(wet_trailer=ROOT/'output/weather_round4/photos/w4_021.jpg',harrow=ROOT/'output/hierarchy_review/f067a0be65404cdea89f113e9e206f92/input.jpg',multiple=ROOT/'output/continuation_round6/web_verification/6345b13e171f419ab19301aed4892993/input.jpg')
    for key,path in cases.items():
        with Image.open(path) as h:im=ImageOps.exif_transpose(h).convert('RGB')
        rows[key]=dict(path=str(path),regions=detect(im,d),semantic=semantic(im,d))
        (OUT/'semantic_diagnosis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');print(key,rows[key]['semantic'],flush=True)
if __name__=='__main__':main()
