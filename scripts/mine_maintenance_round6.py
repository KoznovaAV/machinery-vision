"""Mine training-only implement proposals for visual review; never auto-label them."""
import json,hashlib
from pathlib import Path
import torch
from PIL import Image,ImageOps,ImageDraw
from scripts.predict_machinery_hierarchy import detector_model,attachment_semantics
from src.models.machinery_regions import iou
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def main():
    torch.set_num_threads(5);data=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());det=detector_model()
    rows=sorted([r for r in data['train'] if r['model'] in ('claas_axion_800','mtz_1221','mtz_1523','acros_550','don_1500','dt_75')],key=lambda r:r['sha256'])[:75]
    result=[];pending={r['key'] for r in rows};cache=OUT/'mined_candidates.json'
    if cache.exists():result=json.loads(cache.read_bytes());pending-={r['parent_key'] for r in result}
    for n,row in enumerate(rows):
        if row['key'] not in pending:continue
        with Image.open(ROOT/row['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB')
        raw=attachment_semantics(im,det);selected=[]
        for proposal in sorted(raw,key=lambda r:r['score'],reverse=True):
            label=proposal['label']
            if 'trailer' not in label or proposal['score']<.25:continue
            if any(iou(proposal['box'],v['box'])>.6 for v in selected):continue
            selected.append(proposal)
        for j,v in enumerate(selected):
            box=[max(0,int(v['box'][0])),max(0,int(v['box'][1])),min(im.width,int(v['box'][2])+1),min(im.height,int(v['box'][3])+1)]
            if min(box[2]-box[0],box[3]-box[1])<32:continue
            key='mine6/'+row['key']+'/'+str(j);dest=OUT/'mined_views'/(hashlib.sha256(key.encode()).hexdigest()+'.jpg');dest.parent.mkdir(exist_ok=True);im.crop(box).save(dest,quality=95)
            result.append(dict(key=key,parent_key=row['key'],group=row['group'],parent_path=row['path'],parent_sha256=row['sha256'],path=dest.relative_to(ROOT).as_posix(),bbox=box,score=v['score'],detector_label=label,decision='unreviewed'))
        cache.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        if n%10==0:print('training photos',n+1,len(rows),'proposals',len(result),flush=True)
    for offset in range(0,len(result),24):
        batch=result[offset:offset+24];sheet=Image.new('RGB',(1200,220*((len(batch)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(batch):
            with Image.open(ROOT/r['path']) as h:thumb=ImageOps.contain(h.convert('RGB'),(292,185))
            x=i%4*300;y=i//4*220;sheet.paste(thumb,(x,y));draw.text((x+4,y+190),str(offset+i)+' '+r['parent_key'][-28:],fill='black')
        sheet.save(OUT/('mined_review_%d.jpg'%offset),quality=92)
    print('review candidates',len(result),flush=True)
if __name__=='__main__':main()
