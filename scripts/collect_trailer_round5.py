import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/trailer_round5'
items=[
('https://www.farmersjournal.ie/machinery/news/2016-john-deere-6130m-fetches-59-900-at-kilkenny-dispersal-auction-812585','https://d3mdtxxgfz6upn.cloudfront.net/WEBFILES/000/812/585/2166358-812585.jpg'),
('https://www.jmwood.com/NEW-HOLLAND-T5050-4X4-FARM-TRACTOR-S-N-Z8JH16007-08-YR-3-PTH-PTO-3-REMOTES-ECAB-W-AIR-16-9R30_i19140947','https://dygtyjqp7pi0m.cloudfront.net/i/18787/19140947_1.jpg?v=8D10C790BABB890'),
('https://www.olx.pl/d/oferta/new-holland-t5060-ladowacz-CID757-ID19nVs1.html','https://ireland.apollo.olxcdn.com/v1/files/ngvcu22xrh1j-PL/image%3Bs%3D1024x768'),
('https://www.truck1.lt/zemes-ukio-technika/traktoriai/new-holland-t5050-a8544079.html','https://www.truck1.lt/img/xxl/17257/New-Holland-T-5050-Italija_17257_7067477256796.jpg'),
('https://www.agrofoto.pl/forum/gallery/image/72432-new-holland-t5050/','https://img.agrofoto.pl/monthly_2018_11/new-holland-t5050_4000_71_469560.jpg.9996b9e0c97fc066923f8cf42e8a6ba9.jpg'),
('https://www.tractorhouse.com/listing/for-sale/240307879/2011-new-holland-t5070-100-hp-to-174-hp-tractors','https://media.sandhills.com/img.axd?c=True&checksum=PLRwmDYk8vl%2BcLE%2BfAiVCZC41fW%2FKXRlJSxm6uq4U1o%3D&ext=&h=460&id=9035374784&lp=&p=&rt=0&rwl=False&sz=Max&t=&w=614&wid=4326205933&wt=False'),
('https://www.tractorhouse.com/listing/for-sale/252113813/2013-new-holland-t5070-100-hp-to-174-hp-tractors','https://media.sandhills.com/img.axd?c=True&checksum=bO918LVf1Qik22vRgLvTNewHwdpeaFVqL4DLAeIVMTg%3D&ext=&h=460&id=10028701397&lp=&p=&rt=0&rwl=False&sz=Max&t=&w=614&wid=4326205933&wt=False'),
('https://www.agropool.ch/de/tracteur-new-holland-t5070/a490998/','https://www.agropool.ch/static/ams-prod/Je0ANYE890OLxTNERR00Qg.webp'),
('https://www.njuskalo.hr/traktori/traktor-new-holland-t5070-oglas-38477598','https://www.njuskalo.hr/image-w920x690/traktori/traktor-new-holland-t5070-slika-179741462.jpg'),
('https://www.blinto.se/auction/NEW-Holland-5070-163767-68692/','https://static.blinto.se/object/image/big/big_163767_1317903.jpg?time=1680252386'),
('https://e-farm.com/en/used-farm-machinery/tractors/new-holland/t5060/hnxmnd8/','https://ik.imagekit.io/efarm/images/098b3039-68f2-4fc6-b7b3-b5f287694065.jpg?tr=w-400%2Cl-image%2Ci-%40%40website-machine-images-watermarks%40%40watermark_DZDDMXPYs_M9F2mQxfH.png%2Clx-5%2Cly-5%2Cw-60%2Cl-end'),
('https://e-farm.com/en/used-farm-machinery/tractors/new-holland/t5060/ztessnu/','https://ik.imagekit.io/efarm/images/ad66cdd2-c1a3-409e-9792-8fd7f6c308e7.jpg?tr=w-600%2Car-1000-667%2Cl-image%2Ci-%40%40website-machine-images-watermarks%40%40watermark_DZDDMXPYs_M9F2mQxfH.png%2Clx-6%2Cly-6%2Cw-90%2Cl-end')]
rows=[dict(id='t5_%03d'%i,source=s,url=u,target='t5000_candidates',license='unverified') for i,(s,u) in enumerate(items)]
(OUT/'queue.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
import scripts.collect_mvp_photos as c
c.OUT=OUT;c.main()
