"""Validate proposed release on frozen OCR, then full winter/user regressions."""
import importlib.util,json
from pathlib import Path
import numpy as np
from scripts.train_weather_round4 import evaluate
from scripts.train_weather_round4_specialist import load_vectors,count
from src.models.hierarchy import predict_hierarchy
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output/weather_round4'

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes()); markings=json.loads((OUT/'control_markings.json').read_bytes())
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes()); candidate=json.loads((OUT/'nh_only_candidate.json').read_bytes())
    full,crop=load_vectors(); z=np.load(OUT/'features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):(full if str(k).startswith('full/') else crop)[str(k)[5:]]=v
    vv={k:np.concatenate((v,crop[k])) for k,v in full.items() if k in crop}
    spec=importlib.util.spec_from_file_location('round4_old_hierarchy',OUT/'baseline/hierarchy_source.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    import scripts.train_weather_round4 as training
    comparison={}
    for name,bundle,fn in [('before',baseline,old.predict_hierarchy),('after',candidate,predict_hierarchy)]:
        training.predict_hierarchy=fn
        comparison[name]={s:evaluate(bundle,data[s],vv,markings) for s in ('test','winter_test','fresh_test','new_photos_test')}
    oldc=comparison['before']['test']['by_class']; newc=comparison['after']['test']['by_class']
    assert all(newc[k]['family_correct']>=v['family_correct'] and newc[k]['label_correct']>=v['label_correct'] and newc[k]['correct_answers']>=v['correct_answers'] and newc[k]['wrong_answers']<=v['wrong_answers'] for k,v in oldc.items())
    (OUT/'release_ocr_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf8')
    print('OCR release', {n:{s:{k:r[k] for k in ('family_correct','answers','wrong_answers','correct_answers')} for s,r in ss.items()} for n,ss in comparison.items()},flush=True)
    import scripts.predict_machinery_hierarchy as pipeline
    cases={r['key']:ROOT/r['path'] for r in data['winter_test']}
    cases.update(mud=ROOT/'output/hierarchy_review/c7f689a77538495b893703dd765ca5b8/input.jpg',attachment=ROOT/'output/hierarchy_review/f067a0be65404cdea89f113e9e206f92/input.jpg',multiple=ROOT/'output/continuation_round6/web_verification/6345b13e171f419ab19301aed4892993/input.jpg',new_wet=ROOT/'output/weather_round4/photos/w4_021.jpg')
    summary={}
    for name,bundle_path,fn in [('before',OUT/'baseline/hierarchy_bundle.json',old.predict_hierarchy),('after',OUT/'nh_only_candidate.json',predict_hierarchy)]:
        pipeline.predict_hierarchy=fn; summary[name]={}
        for key,path in cases.items():
            dest=OUT/'pipeline'/name/key.replace('/','_')/'prediction.json'
            result=pipeline.run(path,bundle_path,dest)
            rows=[{k:o.get(k) for k in ('family_candidate','family','visible_model_marking','candidate_source','condition','condition_reason','quality','reason','type','type_candidate')} for o in result.get('objects',[])]
            summary[name][key]=dict(objects=rows,ignored=len(result.get('ignored_regions',[])))
            print(name,key,[(r['family_candidate'],r['family'],r['condition'],r['visible_model_marking']) for r in rows],flush=True)
            (OUT/'pipeline_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
