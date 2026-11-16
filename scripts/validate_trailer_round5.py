"""Compare full pipelines before/after the reference-confirmed trailer filter."""
import importlib.util,json,hashlib
from pathlib import Path
import scripts.predict_machinery_hierarchy as current
from src.models.mvp_inference import verify_trailer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def main():
    spec=importlib.util.spec_from_file_location('trailer5_before',OUT/'baseline/predict_source.py');before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before);before.ROOT=ROOT
    bundle=json.loads((OUT/'reference_region_candidate.json').read_bytes());current.verify_trailer=lambda value:verify_trailer(value,bundle=bundle)
    audit=json.loads((OUT/'region_control_audit.json').read_bytes());data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    cases={key:ROOT/r['path'] for key,r in audit.items() if r['ignored']}
    cases.update({r['key']:ROOT/r['path'] for r in data['round5_test']})
    cases.update({r['key']:ROOT/r['path'] for r in json.loads((OUT/'new_region_test.json').read_bytes())})
    cases.update(mud=ROOT/'output/hierarchy_review/c7f689a77538495b893703dd765ca5b8/input.jpg',attachment=ROOT/'output/hierarchy_review/f067a0be65404cdea89f113e9e206f92/input.jpg',multiple=ROOT/'output/continuation_round6/web_verification/6345b13e171f419ab19301aed4892993/input.jpg')
    for name in ('m004','m011','m028','m000'):cases['mvp/'+name]=ROOT/'output/mvp/photos'/(name+'.jpg')
    summary={};source_hash=hashlib.sha256((ROOT/'scripts/predict_machinery_hierarchy.py').read_bytes()).hexdigest()
    for mode,predictor in [('before',before),('after',current)]:
        summary[mode]={}
        for key,path in cases.items():
            dest=OUT/'pipeline'/mode/key.replace('/','_')/'prediction.json'; dest.parent.mkdir(parents=True,exist_ok=True)
            cached=ROOT/'output/weather_round4/pipeline/after'/key.replace('/','_')/'prediction.json'
            if mode=='before' and dest.exists():result=json.loads(dest.read_bytes())
            elif mode=='before' and cached.exists():result=json.loads(cached.read_bytes());dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
            else:result=predictor.run(path,OUT/'baseline/hierarchy_bundle.json',dest)
            summary[mode][key]=dict(path=str(path),result=str(dest),objects=[{k:r.get(k) for k in ('family_candidate','family','visible_model_marking','condition','image_quality','vehicle_type')} for r in result.get('objects',[])],ignored=[r.get('reason') for r in result.get('ignored_regions',[])])
            (OUT/'pipeline_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
            print(mode,key,len(summary[mode][key]['objects']),summary[mode][key]['ignored'],flush=True)
    after=summary['after']
    for key,value in after.items():
        if key!='weather4/w4_021':assert value['objects']==summary['before'][key]['objects'],key
    assert len(after['weather4/w4_021']['objects'])==1 and 'semantic_trailer' in after['weather4/w4_021']['ignored']
    assert after['weather4/w4_021']['objects'][0]['family'] is None
    assert len(after['model_wave2/tractorbg-293984']['objects'])==len(summary['before']['model_wave2/tractorbg-293984']['objects'])
    assert len(after['round5/r5-143']['objects'])==len(summary['before']['round5/r5-143']['objects'])
    assert len(after['multiple']['objects'])==3 and len(after['attachment']['objects'])==1
    assert after['mud']['objects'][0]['condition']=='dirty'
    assert after['mvp/m004']['objects'][0]['image_quality']['low_light']
    for key in ('mvp/m011','mvp/m028'):assert after[key]['objects'][0]['visible_model_marking']==summary['before'][key]['objects'][0]['visible_model_marking']
    (OUT/'pipeline_checks.json').write_text(json.dumps(dict(passed=True,source_sha256=source_hash,cases=len(cases),note='Known wet trailer is feedback training. Two newly collected trailer sources and T5060 are held out.'),indent=2),encoding='utf8')
    print('Full pipeline checks passed',flush=True)
if __name__=='__main__':main()
