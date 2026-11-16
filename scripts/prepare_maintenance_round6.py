"""Manually reviewed mining results, with parent groups preserved."""
import json,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def main():
    rows=json.loads((OUT/'mined_candidates.json').read_bytes())
    negatives={2,3,21,23,28,29,30,45,48,52,56}
    positives={0,1,4,6,9,11,16,17,19,20,22,24,26,27,32,33,34,35,36,37,38,39,40,41,42,44,46,49,50,53,54,55}
    for i,r in enumerate(rows):
        r['decision']='accepted' if i in negatives|positives else 'rejected_ambiguous'
        r['manual_label']='attachment' if i in negatives else 'machine' if i in positives else None
        r['manual_note']='Reviewed crop and parent image; keep this parent source in training only.' if i in negatives|positives else 'Insufficient view or ambiguous object; do not train.'
    downloaded=json.loads((OUT/'downloaded.json').read_bytes());external=[]
    for r in downloaded:
        if r['status']!='downloaded':continue
        if r['id']!='samson_04':continue
        p=OUT/'photos'/(r['id']+'.jpg')
        external.append(dict(r,key='maint6/'+r['id'],path=p.relative_to(ROOT).as_posix(),group='source-'+hashlib.sha256(r['source'].encode()).hexdigest()[:16],model='attachment',decision='accepted',manual_note='Real trailed Samson slurry tanker, no cab or independent engine; tractor at far right is excluded by training crop.',crop_fraction=[0,0,.90,1]))
    (OUT/'manual_region_review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'manual_external_review.json').write_text(json.dumps(external,ensure_ascii=False,indent=2),encoding='utf8')
    print('mined accepted',len(negatives),'attachments',len(positives),'machine fragments; external',len(external),flush=True)
if __name__=='__main__':main()
