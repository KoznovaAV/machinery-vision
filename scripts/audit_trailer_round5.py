"""Audit region veto on unchanged recognition controls before releasing it."""
import json
from pathlib import Path
import torch
from PIL import Image,ImageOps,ImageDraw
from scripts.predict_machinery_hierarchy import detector_model,detect,attachment_semantics
from src.models.trailer_filter import split_trailers
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def main():
    torch.set_num_threads(5);d=detector_model();data=json.loads((ROOT/'output/weather_round4/dataset_manifest.json').read_bytes())
    rows=[dict(r,split=s) for s in ('test','winter_test','fresh_test','new_photos_test','round4_test') for r in data[s]]
    result_path=OUT/'region_control_audit.json';cache=json.loads(result_path.read_bytes()) if result_path.exists() else {}
    for n,row in enumerate(rows):
        if row['key'] in cache:continue
        with Image.open(ROOT/row['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB')
        regions=detect(im,d);semantic=attachment_semantics(im,d) if len(regions)>1 else []
        kept,ignored=split_trailers(regions,semantic)
        cache[row['key']]=dict(path=row['path'],split=row['split'],model=row.get('model'),before=regions,after=kept,ignored=ignored,semantic=semantic)
        result_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf8')
        if ignored:
            canvas=im.copy();draw=ImageDraw.Draw(canvas)
            for r in kept:draw.rectangle(r['box'],outline='green',width=5)
            for r in ignored:draw.rectangle(r['bbox'],outline='red',width=5)
            canvas.thumbnail((1000,800));canvas.save(OUT/('veto_'+row['key'].replace('/','_')+'.jpg'),quality=90)
            print('VETO',row['key'],len(regions),'->',len(kept),flush=True)
        if n%20==0:print('regions',n+1,len(rows),flush=True)
    print('audited',len(cache),'changed',[k for k,v in cache.items() if v['ignored']],flush=True)
if __name__=='__main__':main()
