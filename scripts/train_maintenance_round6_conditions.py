"""Evaluate new condition heads on unchanged grouped condition controls."""
import copy,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps
from scripts.train_trailer_round5_verifier import vectors
from scripts.train_mvp import train
from scripts.train_machinery_hierarchy import calibrate
from scripts.train_winter_round2 import condition_benchmark
from src.models.condition_features import weather_measurements
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/maintenance_round6'
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    full,crop=vectors();data=json.loads((ROOT/'output/trailer_round5/dataset_manifest.json').read_bytes());baseline=json.loads((OUT/'baseline/condition_bundle.json').read_bytes());held=[r for r in data['condition_rows'] if r['role']=='test'];pixel={}
    for r in data['condition_rows']:
        with Image.open(ROOT/r['path']) as h:im=ImageOps.exif_transpose(h).convert('RGB')
        pixel[r['key']]=np.concatenate((full[r['key']],weather_measurements(im)*5))
    trials={'baseline':baseline};training={}
    for attribute in ('dirty','snow','rain','wet'):
        rows=[dict(r,model=str(r['condition'][attribute])) for r in data['condition_rows'] if r['role']=='train' and r['condition'].get(attribute) is not None];groups={label:len({r['group'] for r in rows if r['model']==label}) for label in ('0','1')}
        assert not {r['group'] for r in rows}&{r['group'] for r in held}
        if min(groups.values())<6:training[attribute]=dict(skipped=True,groups=groups);continue
        options={'crop':crop,'scene':full} if attribute=='dirty' else {'scene_weather':pixel} if attribute=='snow' else {'scene':full,'crop':crop}
        candidates={name:train(rows,v,min_groups=10 if attribute=='dirty' else 6) for name,v in options.items()};source=max(candidates,key=lambda name:candidates[name][1]['grouped_oof_balanced_accuracy']);head,summary=candidates[source];cal=calibrate(summary['oof_records'],head['classes'],min_groups=10 if attribute=='dirty' else 6,target_precision=.98)
        for value in cal.values():
            if value['threshold'] is not None:value['threshold']=max(.95,value['threshold'])
        candidate=copy.deepcopy(baseline);candidate['heads'][attribute]=head;candidate['calibration'][attribute]=cal;candidate.setdefault('feature_sources',{})[attribute]=source
        trials[attribute]=candidate;training[attribute]=dict(source=source,groups=groups,metrics=summary,calibration=cal)
        save(attribute+'_condition_candidate.json',candidate);print(attribute,groups,'source',source,'cal',cal,flush=True)
    results={name:condition_benchmark(b,held,full,crop) for name,b in trials.items()};gates={}
    for name in trials:
        if name=='baseline':continue
        a=results['baseline'][name];b=results[name][name];gates[name]=dict(no_more_errors=b['wrong_answers']<=a['wrong_answers'],answers_preserved=b['answers']-b['wrong_answers']>=a['answers']-a['wrong_answers'],groups_preserved=b['correct_positive_groups']>=a['correct_positive_groups'])
    save('condition_comparison.json',results);save('condition_training.json',training);save('condition_gates.json',gates)
    print('RESULTS',{n:{a:{k:v[k] for k in ('total','answers','wrong_answers','positive_answers')} for a,v in rr.items()} for n,rr in results.items()},flush=True);print('GATES',gates,flush=True)
if __name__=='__main__':main()
