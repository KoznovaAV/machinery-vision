"""Preserve separate review assets per case from the recorded detector boxes."""
import json
from pathlib import Path
from PIL import Image,ImageOps
from scripts.benchmark_tractor_localization import crop
from scripts.benchmark_front_axle import front_region
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'
def main():
    summary=json.loads((OUT/'pipeline_comparison.json').read_bytes())
    for name,cases in summary.items():
        for key in cases:
            old=OUT/'pipeline'/name/(key.replace('/','_')+'.json')
            result=json.loads(old.read_bytes()); dest=OUT/'pipeline'/name/key.replace('/','_')/'prediction.json';dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(result['image']) as handle:image=ImageOps.exif_transpose(handle).convert('RGB')
            for i,obj in enumerate(result['objects']):
                folder=dest.parent/'objects'/str(i);folder.mkdir(parents=True,exist_ok=True)
                machine=crop(image,obj['bbox']); machine.save(folder/'machine.jpg',quality=95);obj['crop']=str((folder/'machine.jpg').resolve())
                details=obj.get('details',{}).get('front_axle')
                if isinstance(details,dict) and details.get('image'):
                    front_region(machine,details['orientation_proposal']).save(folder/'front_proposal.jpg',quality=95);details['image']=str((folder/'front_proposal.jpg').resolve())
            result['review_assets_note']='Per-case JPEGs reconstructed from recorded detector boxes; recognition outputs unchanged.'
            dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Per-case review assets preserved',flush=True)
if __name__=='__main__':main()
