"""Extract frozen RGB features for newly reviewed real photographs only."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
import scripts.build_winter_features as b
from scripts.benchmark_tractor_localization import crop,load_detector,tractor_box
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/robustness_round3'

def main():
    torch.set_num_threads(5)
    data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    rows=[r for r in data['new_review'] if r['decision']=='accepted']
    b.CROP_DIRS=('output/robustness_round3/views',)+b.CROP_DIRS
    needed=[r for r in rows if b.crop_path(r) is None]
    if needed:
        detector=load_detector(ROOT/'weights/grounding-dino-tiny')
        for n,r in enumerate(needed):
            with Image.open(ROOT/r['path']) as handle:image=ImageOps.exif_transpose(handle).convert('RGB')
            found=tractor_box(image,detector,320,544).get('foreground')
            machine=crop(image,found['box']) if found else image
            path=OUT/'views'/r['sha256']/'machine.jpg';path.parent.mkdir(parents=True,exist_ok=True);machine.save(path,quality=95)
            print('crop',n+1,len(needed),r['id'],flush=True)
        del detector
    files={prefix+r['key']:path for r in rows for prefix,path in (('full/',ROOT/r['path']),('crop/',b.crop_path(r)))}
    signature=hashlib.sha256('|'.join(k+':'+hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sorted(files.items())).encode()).hexdigest()
    path=OUT/'features.npz'
    if path.exists() and str(np.load(path)['signature'])==signature:return
    values=b.fused(files);np.savez_compressed(path,signature=signature,keys=list(values),values=list(values.values()))
    print('Features',len(values),flush=True)

if __name__=='__main__':main()
