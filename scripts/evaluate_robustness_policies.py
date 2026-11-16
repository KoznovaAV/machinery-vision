"""Compare refusal policies on cached single-object features, without OCR claims."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scripts.train_winter_round2 import vectors
from scripts.build_winter_features import crop_path
from src.models.hierarchy import predict_hierarchy
from src.models.mvp_inference import assess_conditions
from src.models.image_visibility import assess_visibility,withhold_unreliable_details
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/robustness_round3'

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes());full,crop=vectors()
    z=np.load(OUT/'features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    bundles={'baseline':json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes()),'selected':json.loads((OUT/'combined_candidate.json').read_bytes())}
    states=json.loads((OUT/'baseline/condition_bundle.json').read_bytes());output={n:{} for n in bundles}
    import scripts.build_winter_features as feature_paths
    feature_paths.CROP_DIRS=('output/robustness_round3/views',)+feature_paths.CROP_DIRS
    for split in ('test','winter_test','fresh_test','new_photos_test'):
        records={n:[] for n in bundles}
        for row in data[split]:
            key=row['key'];vector=np.concatenate((full[key],crop[key]))
            with Image.open(ROOT/row['path']) as im:scene=ImageOps.exif_transpose(im).convert('RGB')
            with Image.open(crop_path(row)) as im:machine=ImageOps.exif_transpose(im).convert('RGB')
            visibility=assess_visibility(scene,machine)
            attributes=assess_conditions(full[key],crop[key],scene,bundle=states)
            for name,bundle in bundles.items():
                pred=predict_hierarchy(bundle,vector)
                if attributes['snow'].get('value') is True:pred['family']=None
                if name=='selected':
                    withhold_unreliable_details(pred,dict(attributes),visibility)
                expected=row.get('expected_family')
                if expected is None:
                    from src.config.machinery_taxonomy import family_of
                    expected=family_of(row.get('model'))
                records[name].append(dict(key=key,expected=expected,family=pred['family'],low_light=visibility['low_light']))
        for name,rr in records.items():
            output[name][split]=dict(total=len(rr),answers=sum(r['family'] is not None for r in rr),wrong_answers=sum(r['family'] is not None and r['family']!=r['expected'] for r in rr),low_light=sum(r['low_light'] for r in rr),rows=rr)
    (OUT/'visibility_policy_comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print({n:{s:{k:v[k] for k in ('answers','wrong_answers','low_light')} for s,v in ss.items()} for n,ss in output.items()},flush=True)

if __name__=='__main__':main()
