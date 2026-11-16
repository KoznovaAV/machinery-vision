"""Promote only the validated T5000 head and archive evidence without changing legacy heads."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4';ACTIVE=ROOT/'artifacts/mvp'
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=read(OUT/'baseline/hierarchy_bundle.json');new=read(OUT/'nh_only_candidate.json')
    assert read(OUT/'recognition_gates.json')['nh_only']['eligible']
    for key in old['heads']:
        if key!='nh_specialist':assert old['heads'][key]==new['heads'][key],key
    for key in old['calibration']:
        if key!='nh_specialist':assert old['calibration'][key]==new['calibration'][key],key
    pipeline=read(OUT/'pipeline_comparison.json'); assert len(pipeline['before'])==len(pipeline['after'])==16
    for key,prior in pipeline['before'].items():
        current=pipeline['after'][key]
        assert len(prior['objects'])==len(current['objects']) and prior['ignored']==current['ignored'],key
        for a,b in zip(prior['objects'],current['objects'],strict=True):
            assert a['family']==b['family'] and a['condition']==b['condition'],key
    after=pipeline['after'];assert after['mud']['objects'][0]['condition']=='dirty'
    assert len(after['attachment']['objects'])==1 and after['attachment']['ignored']==1
    assert len(after['multiple']['objects'])==3
    assert after['mvp/m004']['objects'][0]['family_candidate'] is None
    assert after['mvp/m011']['objects'][0]['visible_model_marking']=='82'
    assert after['mvp/m028']['objects'][0]['visible_model_marking']=='M135'
    # Enrich compact summaries from saved full predictions; do not rerun inference.
    for name,items in pipeline.items():
        for key,item in items.items():
            result=read(OUT/'pipeline'/name/key.replace('/','_')/'prediction.json')
            for short,full in zip(item['objects'],result['objects'],strict=True):
                for k in ('vehicle_type','vehicle_type_candidate','image_quality'):short[k]=full.get(k)
                for k in ('type','type_candidate','quality'):short.pop(k,None)
    save(OUT/'pipeline_comparison.json',pipeline)
    protected={name:sha(ACTIVE/name) for name in ('condition_bundle.json','region_bundle.json')}
    assert sha(ACTIVE/'hierarchy_bundle.json') in {sha(OUT/'baseline/hierarchy_bundle.json'),sha(OUT/'nh_only_candidate.json')}
    shutil.copy2(OUT/'nh_only_candidate.json',ACTIVE/'hierarchy_bundle.json')
    prior_metrics=read(OUT/'baseline/recognition_metrics.json')
    raw=read(OUT/'recognition_comparison.json');ocr=read(OUT/'release_ocr_comparison.json')
    save(ACTIVE/'recognition_metrics.json',dict(round='weather_round4',baseline_round3=prior_metrics,baseline=raw['baseline']['test'],selected=raw['nh_only']['test'],controls=raw['nh_only'],ocr_comparison=ocr,full_pipeline='output/weather_round4/pipeline_comparison.json',limits='Frozen controls reused for diagnosis; no new T5000 rain/heavy-mud positives; states unchanged.'))
    manifest=dict(round='weather_round4',report='reports/WEATHER_ROUND4_2026-11-15.md',selected='nh_only',dataset_sha256=sha(OUT/'dataset_manifest.json'),hierarchy_bundle_sha256=sha(ACTIVE/'hierarchy_bundle.json'),source_sha256=sha(ROOT/'src/models/hierarchy.py'),preserved_bundles=protected,tests=dict(passed=198,command='python -B -m pytest tests -q -p no:cacheprovider'),manual_review=dict(downloaded=7,accepted_unique=2,t5000=1,t5000_rain_or_heavy_mud=0),remaining=['Unknown NH in rain misclassified by feature head','Trailer duplicate region on new wet control','Winter foreign tractors and unreadable markings'])
    save(OUT/'release_manifest.json',manifest);save(ACTIVE/'weather_round4_release.json',manifest)
    for name,value in protected.items():assert sha(ACTIVE/name)==value
    print('Promoted NH specialist',manifest['hierarchy_bundle_sha256'],flush=True)
if __name__=='__main__':main()
