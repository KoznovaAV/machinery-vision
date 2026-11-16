import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import scripts.collect_mvp_photos as c
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6';c.OUT=OUT
items=[
('https://www.joskin.com/nl/voorraad-tweedehands/R735','https://quote.joskin.com/system/machine_images/images/000/003/020/original/R735-2.jpg'),
('https://www.joskin.com/en/stock-second-hand/j6549','https://quote.joskin.com/system/machine_images/images/000/000/980/original/J6549.jpg'),
('https://www.joskin.com/en/slurry-tankers/tetraliner','https://quote.joskin.com/system/medias/files/000/000/734/thumb/Tetraliner-7-3000x2000-min.jpg'),
('https://pol-agra.com.pl/produkty/joskin/euroliner-detail','https://pol-agra.com.pl/images/stories/virtuemart/product/Euroliner%2024000%205.jpg'),
('https://www.velatorg.is/notadar-haugsugur/joskin-volumetra-20000d-arg-2022%2C340/','https://www.velatorg.is/user_uploads/cache/14/d6/14d66a5dbab46576e6b7c6e1bb3e7797.jpg'),
('https://www.agriland.ie/farming-news/166008/','https://cdn.agriland.ie/uploads/2017/05/John-Deere-Manure-Sensing-system-1024x499.jpg')]
rows=[dict(id='tank_%02d'%i,source=s,url=u,target='tanker_candidates',license='unverified') for i,(s,u) in enumerate(items)];(OUT/'queue.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');c.main()
