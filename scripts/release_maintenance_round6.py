"""Publish only the validated maintenance region verifier; preserve recognition/state heads."""
import hashlib,json,shutil,sys
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6';ACTIVE=ROOT/'artifacts/mvp'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    checks=read(OUT/'pipeline_checks.json');assert checks['passed'] and checks['cases']==20
    assert checks['source_sha256']==sha(ROOT/'scripts/predict_machinery_hierarchy.py')
    assert not any(r['veto'] for r in read(OUT/'fusion_main_control.json'))
    independent=read(OUT/'independent_region_checks.json');assert independent['new_tanker']['veto']
    assert not any(r['veto'] for r in independent['partial_control'] if r['expected']=='machine')
    assert all(not r['exact_duplicate'] and r['nearest_distance']>6 for r in read(OUT/'external_duplicate_audit.json'))
    model=read(OUT/'fusion_region_candidate.json');old=read(OUT/'baseline/region_bundle.json')
    for key,value in old.items():
        if key!='trailer_reference':assert model[key]==value,key
    for filename in ('hierarchy_bundle.json','condition_bundle.json','region_bundle.json'):
        assert sha(ACTIVE/filename)==sha(OUT/'baseline'/filename),filename
    # Retain the old reference file: the still-running previous service can use it until restart.
    new_name='trailer_references_round6.npz';shutil.copy2(OUT/'balanced_trailer_references.npz',ACTIVE/new_name)
    config=model['trailer_reference'];assert config['sha256']==sha(ACTIVE/new_name)
    config['path']='artifacts/mvp/'+new_name
    save(ACTIVE/'region_bundle.json',model)
    comparison=read(OUT/'pipeline_comparison.json')
    assert len(comparison['after']['round5/r5b-103']['objects'])==1
    assert comparison['after']['weather_expansion/resumable-photo-627531']['uncertain']==1
    for key in ('model_wave2/tractorbg-293984','round5/r5-143','multiple'):
        assert len(comparison['after'][key]['objects'])==len(comparison['before'][key]['objects'])
    if not (OUT/'baseline/region_control.json').exists():shutil.copy2(ACTIVE/'region_control.json',OUT/'baseline/region_control.json')
    metrics=read(OUT/'baseline/region_control.json');metrics['maintenance_round6']=dict(report='reports/MAINTENANCE_ROUND6_2026-11-16.md',pipeline='output/maintenance_round6/pipeline_comparison.json',main_control='output/maintenance_round6/fusion_main_control.json',uncertain_regions_are_not_confirmed_machines=True,independent_region_checks='output/maintenance_round6/independent_region_checks.json')
    save(ACTIVE/'region_control.json',metrics)
    sources=['scripts/predict_machinery_hierarchy.py','scripts/render_hierarchy_review.py','scripts/serve_hierarchy_review.py','scripts/check_mvp_release.py','src/models/mvp_inference.py','src/models/hierarchy.py','src/models/trailer_filter.py','src/models/image_visibility.py','src/models/condition_features.py','src/models/visible_condition.py','src/config/machinery_taxonomy.py','src/config/hierarchy_artifacts.py','configs/machinery_taxonomy.json','scripts/benchmark_tractor_localization.py','scripts/benchmark_grouped_recognition.py','scripts/benchmark_front_axle.py','scripts/analyze_model_marking.py','scripts/predict_tractor_conditions.py','requirements-hierarchy.txt']
    fingerprints={}
    for name in sources:
        fingerprints[name]=sha(ROOT/name);dest=OUT/'release_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    packages={}
    for name in ('torch','torchvision','transformers','timm','numpy','scikit-learn','Pillow','rapidocr-onnxruntime','fastapi','uvicorn'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    save(OUT/'runtime_versions.json',dict(python=sys.version.split()[0],packages=packages))
    release=dict(round='maintenance_round6',release_date='2026-11-16',timezone='Europe/Moscow',report='reports/MAINTENANCE_ROUND6_2026-11-16.md',handover='docs/HANDOVER.md',artifacts={n:sha(ACTIVE/n) for n in ('hierarchy_bundle.json','condition_bundle.json','region_bundle.json',new_name)},sources=fingerprints,tests=222,pipeline_cases=20,preserved=['family/type/MTZ classifier weights','condition weights','original region and partial heads'],condition_candidates_rejected=True,release_source='output/maintenance_round6/release_source',limitations=['Repeated diagnostic controls are not blinded validation','Uncertain regions require human review','Rain and machine wetness remain insufficiently validated'],automatic_training=False,automatic_deployment=False)
    save(OUT/'release_manifest.json',release);save(ACTIVE/'maintenance_round6_release.json',release);save(ACTIVE/'current_release.json',dict(round=release['round'],manifest='artifacts/mvp/maintenance_round6_release.json'))
    assert sha(ACTIVE/'hierarchy_bundle.json')==sha(OUT/'baseline/hierarchy_bundle.json')
    assert sha(ACTIVE/'condition_bundle.json')==sha(OUT/'baseline/condition_bundle.json')
    print('Released maintenance region verifier. Family/state weights unchanged.',flush=True)
if __name__=='__main__':main()
