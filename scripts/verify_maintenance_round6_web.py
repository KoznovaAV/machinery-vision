"""Verify installed release over HTTP and assert inference never mutates weights."""
import hashlib,json,re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'artifacts/mvp/maintenance_round6_release.json').read_bytes());before={n:sha(ROOT/'artifacts/mvp'/n) for n in manifest['artifacts']}
    suite=json.loads((OUT/'pipeline_comparison.json').read_bytes())['after'];mapping=dict(tanker='round5/r5b-103',uncertain_trailer='weather_expansion/resumable-photo-627531',real_combine='model_wave2/tractorbg-293984',multiple='multiple',mud='mud',night='mvp/m004');records={};session=requests.Session();session.get('http://127.0.0.1:7864/',timeout=15).raise_for_status()
    for name,key in mapping.items():
        source=Path(suite[key]['path'])
        with source.open('rb') as handle:response=session.post('http://127.0.0.1:7864/predict',files={'file':(source.name,handle,'image/jpeg')},timeout=180)
        response.raise_for_status();(OUT/('web_'+name+'.html')).write_text(response.text,encoding='utf8');folder=re.search(r'/artifacts/([a-f0-9]+)/',response.text).group(1);p=ROOT/'output/hierarchy_review'/folder/'prediction.json';result=json.loads(p.read_bytes());objects=result['objects'];assert len(objects)==len(suite[key]['objects']);assert result['sha256']==sha(source)
        if name=='tanker':
            assert len(objects)==1 and 'composite_duplicate' in [r['reason'] for r in result['ignored_regions']]
            ev=next(r['image_reference'] for r in result['ignored_regions'] if r['reason']=='semantic_trailer')
            assert ev['supplement_label']=='attachment' and ev['supplement_score']>=.95
        if name=='uncertain_trailer':
            assert len(objects)==1 and len(result['uncertain_regions'])==1 and 'Неподтверждённая область' in response.text
            assert objects[0]['condition']=='not_assessed' and 'спорного оборудования' in objects[0]['condition_reason']
            assert '<h2>Машина 2</h2>' not in response.text
        if name=='real_combine':assert len(objects)==3
        if name=='multiple':assert len(objects)==3
        if name=='mud':assert objects[0]['condition']=='dirty'
        if name=='night':assert objects[0]['image_quality']['low_light'] and objects[0]['condition']=='not_assessed'
        records[name]=dict(source=str(source),result=str(p),objects=len(objects),uncertain=len(result.get('uncertain_regions',[])),ignored=[r['reason'] for r in result.get('ignored_regions',[])],http=response.status_code)
        (OUT/'live_web_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8');print(name,'verified',len(objects),flush=True)
    after={n:sha(ROOT/'artifacts/mvp'/n) for n in manifest['artifacts']};assert before==after
    (OUT/'inference_immutability.json').write_text(json.dumps(dict(verified=True,requests=len(records),artifacts=after),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
