import json,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
from scripts.build_winter_features import fused
from src.models.hierarchy import vote,qualified
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def main():
    torch.set_num_threads(5);rows=json.loads((OUT/'trailer_head_comparison.json').read_bytes());bundle=json.loads((OUT/'region_candidate.json').read_bytes());files={};items=[]
    for r in rows:
        if not r['key'].startswith('region-control/'):continue
        if any(v['key']==r['key'] for v in items):continue
        with Image.open(ROOT/r['path']) as h:im=h.convert('RGB')
        for name,ratio in [('upper',(0,0,1,.65)),('upper_center',(.15,0,.85,.65))]:
            box=tuple(int(v*dim) for v,dim in zip(ratio,(im.width,im.height,im.width,im.height)));p=OUT/'focus_views'/(hashlib.sha256((r['key']+name).encode()).hexdigest()+'.jpg');p.parent.mkdir(exist_ok=True);im.crop(box).save(p,quality=95);files[r['key']+'/'+name]=p
        items.append(r)
    vv=fused(files);np.savez_compressed(OUT/'focus_features.npz',keys=list(vv),values=list(vv.values()))
    for r in items:
        result=[]
        for name in ('upper','upper_center'):
            label,score,_=vote(bundle['trailer_head'],vv[r['key']+'/'+name]);result.append(dict(view=name,label=label,score=score,accepted=label=='attachment' and qualified(bundle['trailer_calibration'],label,score),path=str(files[r['key']+'/'+name])))
        r['focus']=result;print(r['source_key'],[(v['label'],round(v['score'],3),v['accepted']) for v in result],flush=True)
    (OUT/'focus_comparison.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
