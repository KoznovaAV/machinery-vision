"""Release trailer references only after frozen-control and full-pipeline checks."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5';ACTIVE=ROOT/'artifacts/mvp'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    check=read(OUT/'pipeline_checks.json');assert check['passed'] and check['source_sha256']==sha(ROOT/'scripts/predict_machinery_hierarchy.py')
    comparison=read(OUT/'pipeline_comparison.json');assert len(comparison['before'])==len(comparison['after'])==19
    for key,value in comparison['after'].items():
        if key!='weather4/w4_021':assert value['objects']==comparison['before'][key]['objects'],key
    assert not any(r['veto'] for r in read(OUT/'reference_main_control.json'))
    references=read(OUT/'reference_comparison.json');assert not any(r['reference_veto'] for r in references if r.get('model')=='machine')
    held=[r for r in references if r['key'].startswith('trailer5/')];assert len(held)==2 and all(r['reference_veto'] for r in held)
    for name in ('hierarchy_bundle.json','condition_bundle.json','region_bundle.json'):
        assert sha(ACTIVE/name)==sha(OUT/'baseline'/name),name
    model=read(OUT/'reference_region_candidate.json');old=read(OUT/'baseline/region_bundle.json')
    for key,value in old.items():assert model[key]==value,key
    rows=read(OUT/'trailer_head_training.json')['rows'];held_rows=read(OUT/'new_region_test.json');data=read(OUT/'dataset_manifest.json')
    frozen=[r for s in ('test','winter_test','fresh_test','new_photos_test','round5_test') for r in data[s]]
    assert not {r['group'] for r in rows}&{r['group'] for r in frozen+held_rows}
    assert data['round4_test']==[] and data['round4_feedback_regression'][0]['key']=='weather4/w4_021'
    if not (OUT/'baseline/region_control.json').exists():shutil.copy2(ACTIVE/'region_control.json',OUT/'baseline/region_control.json')
    shutil.copy2(OUT/'trailer_references.npz',ACTIVE/'trailer_references.npz')
    model['trailer_reference']['path']='artifacts/mvp/trailer_references.npz'
    assert model['trailer_reference']['sha256']==sha(ACTIVE/'trailer_references.npz')
    save(ACTIVE/'region_bundle.json',model)
    metrics=read(OUT/'baseline/region_control.json');metrics['trailer_round5']=dict(main_control='output/trailer_round5/reference_main_control.json',pipeline='output/trailer_round5/pipeline_comparison.json',reference_comparison='output/trailer_round5/reference_comparison.json',known_feedback='weather4/w4_021',independent_new_trailer_sources=2,main_control_vetoes=0,warning='Known wet trailer is training feedback, not independent validation. Other trailer/tank/blade errors remain.')
    save(ACTIVE/'region_control.json',metrics)
    release=dict(round='trailer_round5',report='reports/TRAILER_ROUND5_2026-11-15.md',dataset_sha256=sha(OUT/'dataset_manifest.json'),artifacts={n:sha(ACTIVE/n) for n in ('hierarchy_bundle.json','condition_bundle.json','region_bundle.json','trailer_references.npz')},sources={n:sha(ROOT/n) for n in ('scripts/predict_machinery_hierarchy.py','scripts/render_hierarchy_review.py','src/models/mvp_inference.py','src/models/trailer_filter.py')},tests=212,pipeline_cases=19,preserved=['family/type/MTZ heads','condition heads','original region and partial heads'],known_feedback=['weather4/w4_021'],new_photos=dict(t5000_dirty_training=4,dirty_t5000_machine_groups=1,t5000_control=1,trailer_sources=5,held_out_trailer_sources=2,t5000_confirmed_new_rain=0))
    save(OUT/'release_manifest.json',release);save(ACTIVE/'trailer_round5_release.json',release)
    assert sha(ACTIVE/'hierarchy_bundle.json')==sha(OUT/'baseline/hierarchy_bundle.json')
    assert sha(ACTIVE/'condition_bundle.json')==sha(OUT/'baseline/condition_bundle.json')
    print('Released trailer verifier; family and condition weights unchanged.',flush=True)
if __name__=='__main__':main()
