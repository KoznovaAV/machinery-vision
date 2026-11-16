"""Compare old/new marking policy using exactly the same OCR and frozen features."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scripts.analyze_model_marking import inspect
from scripts.train_weather_round4_specialist import load_vectors,count
import scripts.build_winter_features as paths
from src.models.hierarchy import predict_hierarchy
from src.config.machinery_taxonomy import family_of
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'

def main():
    from rapidocr_onnxruntime import RapidOCR
    data=json.loads((ROOT/'output/robustness_round3/dataset_manifest.json').read_bytes())
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes());full,crop=load_vectors()
    spec=importlib.util.spec_from_file_location('round4_old_hierarchy',OUT/'baseline/hierarchy_source.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cache=OUT/'control_markings.json';ocr=json.loads(cache.read_bytes()) if cache.exists() else {}
    paths.CROP_DIRS=('output/robustness_round3/views',)+paths.CROP_DIRS
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1)
    results={'before':{},'after':{}}
    for split in ('test','winter_test','fresh_test','new_photos_test'):
        records={name:[] for name in results}
        for i,row in enumerate(data[split]):
            sha=row['sha256'];key=row['key']
            if sha not in ocr:
                with Image.open(paths.crop_path(row)) as handle:machine=ImageOps.exif_transpose(handle).convert('RGB')
                ocr[sha]=inspect(machine,engine)
                cache.write_text(json.dumps(ocr,ensure_ascii=False,indent=2),encoding='utf-8')
            x=np.concatenate((full[key],crop[key]))
            for name,fn in (('before',old.predict_hierarchy),('after',predict_hierarchy)):
                pred=fn(baseline,x,ocr[sha]);expected=row.get('expected_family',family_of(row.get('model')))
                records[name].append(dict(key=key,expected=row.get('model'),expected_family=expected,
                    candidate=pred['family_candidate'],family=pred['family'],label=pred['label_candidate'],
                    visible_marking=pred['visible_model_marking'],marking_conflict=pred['marking_conflict'],candidate_source=pred.get('candidate_source','visual_classifiers')))
            if i%20==0:print('OCR',split,i+1,'/',len(data[split]),flush=True)
        for name,rr in records.items():results[name][split]=dict(**count(rr),by_class={str(label):count([r for r in rr if str(r['expected'])==label]) for label in sorted({str(r['expected']) for r in rr})},rows=rr)
        (OUT/'marking_policy_comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print({n:{s:{k:v[k] for k in ('family_correct','answers','wrong_answers','correct_answers')} for s,v in ss.items()} for n,ss in results.items()},flush=True)
    changed=[]
    for split in results['after']:
        for a,b in zip(results['before'][split]['rows'],results['after'][split]['rows'],strict=True):
            if a['candidate']!=b['candidate'] or a['visible_marking']!=b['visible_marking']:changed.append(dict(split=split,before=a,after=b))
    (OUT/'changed_marking_predictions.json').write_text(json.dumps(changed,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
