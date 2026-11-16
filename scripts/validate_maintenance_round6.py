"""Validate maintenance candidate against the previous local release."""
import json,importlib.util,hashlib
from pathlib import Path
import scripts.predict_machinery_hierarchy as candidate
from src.models.mvp_inference import verify_trailer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def main():
    source_sha=hashlib.sha256((ROOT/'scripts/predict_machinery_hierarchy.py').read_bytes()).hexdigest()
    old=json.loads((ROOT/'output/trailer_round5/pipeline_comparison.json').read_bytes())['after'];bundle=json.loads((OUT/'fusion_region_candidate.json').read_bytes());candidate.verify_trailer=lambda v:verify_trailer(v,bundle=bundle)
    before={k:dict(v) for k,v in old.items()};cases={k:Path(v['path']) for k,v in old.items()};control=json.loads((OUT/'new_tanker_control.json').read_bytes());cases['new_fliegl']=ROOT/control['path']
    spec=importlib.util.spec_from_file_location('maintenance_before',OUT/'baseline/predict_source.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.ROOT=ROOT
    p=OUT/'pipeline/before/new_fliegl/prediction.json';result=prior.run(cases['new_fliegl'],OUT/'baseline/hierarchy_bundle.json',p)
    before['new_fliegl']=summary(result,p,cases['new_fliegl'])
    rr=dict(before=before,after={})
    for key,path in cases.items():
        dest=OUT/'pipeline/after'/key.replace('/','_')/'prediction.json';r=candidate.run(path,OUT/'baseline/hierarchy_bundle.json',dest);rr['after'][key]=summary(r,dest,path)
        (OUT/'pipeline_comparison.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2),encoding='utf8');print(key,len(before[key]['objects']),'->',len(r['objects']),'uncertain',len(r.get('uncertain_regions',[])),'ignored',rr['after'][key]['ignored'],flush=True)
    a=rr['after'];assert len(a['round5/r5b-103']['objects'])==1
    assert 'semantic_trailer' in a['round5/r5b-103']['ignored'] and 'composite_duplicate' in a['round5/r5b-103']['ignored']
    assert len(a['weather_expansion/resumable-photo-627531']['objects'])==1 and a['weather_expansion/resumable-photo-627531']['uncertain']==1
    for key in ('model_wave2/tractorbg-293984','round5/r5-143','multiple'):assert len(a[key]['objects'])==len(before[key]['objects']),key
    assert a['mvp/m004']['objects'][0]['image_quality']['low_light'];assert a['mud']['objects'][0]['condition']=='dirty'
    assert a['mvp/m011']['objects'][0]['visible_model_marking']=='82' and a['mvp/m028']['objects'][0]['visible_model_marking']=='M135'
    assert not any(r['veto'] for r in json.loads((OUT/'fusion_main_control.json').read_bytes()))
    (OUT/'pipeline_checks.json').write_text(json.dumps(dict(passed=True,cases=len(cases),source_sha256=source_sha),indent=2),encoding='utf8')
    print('checks passed',flush=True)
def summary(r,dest,path):return dict(path=str(path),result=str(dest),objects=[{k:o.get(k) for k in ('family_candidate','family','condition','visible_model_marking','image_quality')} for o in r.get('objects',[])],ignored=[o['reason'] for o in r.get('ignored_regions',[])],uncertain=len(r.get('uncertain_regions',[])))
if __name__=='__main__':main()
