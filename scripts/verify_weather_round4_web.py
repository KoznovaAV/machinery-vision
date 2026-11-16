"""Exercise the actual restarted upload service and verify its saved predictions."""
import hashlib,json,re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/weather_round4'
def main():
    session=requests.Session();r=session.get('http://127.0.0.1:7864/',timeout=15);r.raise_for_status()
    cases=dict(marking82=ROOT/'output/mvp/photos/m011.jpg',unsupportedM135=ROOT/'output/mvp/photos/m028.jpg',attachment=ROOT/'output/hierarchy_review/f067a0be65404cdea89f113e9e206f92/input.jpg')
    wanted=hashlib.sha256((ROOT/'artifacts/mvp/hierarchy_bundle.json').read_bytes()).hexdigest();records={}
    for key,path in cases.items():
        with path.open('rb') as f:r=session.post('http://127.0.0.1:7864/predict',files={'file':(path.name,f,'image/jpeg')},timeout=180)
        r.raise_for_status();(OUT/('web_'+key+'.html')).write_text(r.text,encoding='utf8')
        folder=re.search(r'/artifacts/([a-f0-9]+)/',r.text).group(1)
        result_path=ROOT/'output/hierarchy_review'/folder/'prediction.json';result=json.loads(result_path.read_bytes())
        assert result['bundle_sha256']==wanted
        objects=result['objects']
        if key=='marking82':assert objects[0]['visible_model_marking']=='82' and objects[0]['family_candidate']=='mtz_80_family' and objects[0]['family'] is None
        if key=='unsupportedM135':assert objects[0]['visible_model_marking']=='M135' and objects[0]['family_candidate']=='unknown' and objects[0]['family'] is None
        if key=='attachment':assert len(objects)==1 and len(result['ignored_regions'])==1
        records[key]=dict(source=str(path),result=str(result_path),http=r.status_code,bundle_sha256=wanted,objects=len(objects),ignored=len(result['ignored_regions']))
        (OUT/'live_web_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
        print(key,'verified',flush=True)
if __name__=='__main__':main()
