"""Verify actual HTTP upload responses from the restarted trailer release."""
import json,re,hashlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
def main():
    session=requests.Session();session.get('http://127.0.0.1:7864/',timeout=15).raise_for_status()
    comparison=json.loads((OUT/'pipeline_comparison.json').read_bytes())['after']
    mapping=dict(wet_feedback='weather4/w4_021',multiple='multiple',real_combine='model_wave2/tractorbg-293984',standalone_trailer='trailer5/neg_01',night='mvp/m004')
    rows={};family_sha=hashlib.sha256((ROOT/'artifacts/mvp/hierarchy_bundle.json').read_bytes()).hexdigest()
    for name,key in mapping.items():
        path=Path(comparison[key]['path'])
        with path.open('rb') as handle:response=session.post('http://127.0.0.1:7864/predict',files={'file':(path.name,handle,'image/jpeg')},timeout=180)
        response.raise_for_status();(OUT/('web_'+name+'.html')).write_text(response.text,encoding='utf8')
        match=re.search(r'/artifacts/([a-f0-9]+)/',response.text)
        if match:result_path=ROOT/'output/hierarchy_review'/match.group(1)/'prediction.json'
        else:
            candidates=list((ROOT/'output/hierarchy_review').glob('*/prediction.json'));result_path=max(candidates,key=lambda p:p.stat().st_mtime)
        result=json.loads(result_path.read_bytes());assert result['bundle_sha256']==family_sha
        assert result['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        objects=result['objects'];expected=comparison[key]['objects'];assert len(objects)==len(expected)
        if name=='wet_feedback':assert len(objects)==1 and 'Прицеп исключён' in response.text and result['ignored_regions'][0]['reason']=='semantic_trailer'
        if name=='multiple':assert len(objects)==3
        if name=='real_combine':assert len(objects)==3
        if name=='standalone_trailer':assert not objects and 'Обнаружено прицепное' in response.text
        if name=='night':assert objects[0]['image_quality']['low_light'] and objects[0]['condition']=='not_assessed'
        rows[name]=dict(source=str(path),result=str(result_path),objects=len(objects),ignored=[r.get('reason') for r in result.get('ignored_regions',[])],http=response.status_code)
        (OUT/'live_web_verification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
        print(name,'HTTP verified',len(objects),flush=True)
if __name__=='__main__':main()
