from pathlib import Path
import json,requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
rows=json.loads((OUT/'downloaded.json').read_bytes());pending=[r for r in rows if r['status']=='failed' and r['id'] in ['t5_000','t5_001','t5_002','t5_004','t5_005','t5_006','t5_010','t5_011']]
def retry(r):
    x=dict(r)
    try:
        q=requests.get(x['url'],timeout=(5,20));q.raise_for_status()
        with Image.open(BytesIO(q.content)) as im:im.load();assert min(im.size)>=180
        p=OUT/'photos'/(x['id']+'.jpg');p.write_bytes(q.content);x.update(status='downloaded',retry=True)
    except Exception as e:x.update(retry_error=str(e)[:180])
    return x
with ThreadPoolExecutor(max_workers=4) as pool:retries=list(pool.map(retry,pending))
by={r['id']:r for r in retries};rows=[by.get(r['id'],r) for r in rows];(OUT/'downloaded.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
from scripts.collect_mvp_photos import sheets
import scripts.collect_mvp_photos as c
c.OUT=OUT;sheets(rows)
print([(r['id'],r['status'],r.get('retry_error')) for r in rows],flush=True)
