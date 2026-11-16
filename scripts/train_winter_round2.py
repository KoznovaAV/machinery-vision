"""Group-validated weather round; frozen controls are evaluation only."""
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from scripts.diagnose_round6_regression import features
from scripts.refine_mvp import load_vectors
from scripts.train_mvp import fit, train
from scripts.train_machinery_hierarchy import calibrate, serialize
from src.config.machinery_taxonomy import MTZ_SPECIALIST, family_of
from src.models.hierarchy import predict_hierarchy
from src.models.mvp_inference import assess_conditions
from src.models.condition_features import weather_measurements
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/winter_round2'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def vectors():
    full, crop = features()
    for name in ('features.npz','hierarchy_features.npz'):
        z = np.load(ROOT / 'output/continuation_round6' / name)
        for k,v in zip(z['keys'],z['values'],strict=True):
            (full if str(k).startswith('full/') else crop)[str(k)[5:]] = v
    f,c = load_vectors()
    full.update(f); crop.update(c)
    z = np.load(OUT / 'features.npz')
    for k,v in zip(z['keys'],z['values'],strict=True):
        (full if str(k).startswith('full/') else crop)[str(k)[5:]] = v
    return full,crop

def recognition_train(rows, rgb, gray):
    x = np.stack([rgb[r['key']] for r in rows])
    g = np.stack([gray[r['key']] for r in rows])
    y = np.array([r['model'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    classes = np.unique(y)
    folds = list(StratifiedGroupKFold(3,shuffle=True,random_state=911).split(x,y,groups))
    trials = []
    for augment in (False,True):
        for c in (.03,.1,1.):
            p = np.zeros((len(rows),len(classes)))
            for a,b in folds:
                assert not set(groups[a]) & set(groups[b])
                rr = [rows[i] for i in a]
                xx = x[a]
                if augment:
                    rr = rr + [dict(rows[i],key=rows[i]['key']+'/gray') for i in a]
                    xx = np.concatenate((xx,g[a]))
                model = fit(rr,xx,c)
                prob = model.predict_proba(x[b])
                for i,label in enumerate(model.classes_):
                    p[b,list(classes).index(label)] = prob[:,i]
            score = float(balanced_accuracy_score(y,classes[p.argmax(axis=1)]))
            trials.append((score,augment,c,p))
            print('trial',augment,c,score,flush=True)
    score,augment,c,p = max(trials,key=lambda t:(t[0],not t[1],-t[2]))
    rr = rows + [dict(r,key=r['key']+'/gray') for r in rows] if augment else rows
    model = fit(rr,np.concatenate((x,g)) if augment else x,c)
    records = [dict(group=r['group'],expected=r['model'],probabilities=pp.tolist()) for r,pp in zip(rows,p,strict=True)]
    thresholds = calibrate(records,classes,min_groups=10,target_precision=.95)
    return serialize(model),dict(c=c,grayscale_training=augment,grouped_oof_balanced_accuracy=score,
        thresholds=thresholds,groups=len(set(groups)),examples=len(rows),
        candidates=[dict(score=t[0],grayscale=t[1],c=t[2]) for t in trials],oof_records=records)

def benchmark(bundle, rows, rgb):
    records = []
    for r in rows:
        pred = predict_hierarchy(bundle,rgb[r['key']])
        expected = r.get('expected_family',family_of(r.get('model')))
        records.append(dict(key=r['key'],expected_family=expected,expected=r.get('model'),
            candidate=pred['family_candidate'],family=pred['family'],label=pred['label_candidate']))
    def counts(rr):
        return dict(total=len(rr),family_correct=sum(r['candidate']==r['expected_family'] for r in rr),
            answers=sum(r['family'] is not None for r in rr),
            wrong_answers=sum(r['family'] is not None and r['family']!=r['expected_family'] for r in rr),
            label_correct=sum(r['label']==r['expected'] for r in rr if r['expected'] not in (None,'other')))
    return dict(**counts(records),by_class={str(label):counts([r for r in records if r['expected']==label])
        for label in sorted({str(r['expected']) for r in records})},rows=records)

def condition_benchmark(bundle, rows, full, crop):
    reports = {a:[] for a in ('dirty','snow','rain','wet')}
    for r in rows:
        with Image.open(ROOT / r['path']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
        predictions = assess_conditions(full[r['key']],crop[r['key']],image,bundle=bundle)
        for a in reports:
            truth = r['condition'].get(a)
            if truth is None:
                continue
            pred = predictions[a]
            reports[a].append(dict(key=r['key'],group=r['group'],expected=truth,**pred))
    return {a:dict(total=len(rr),positive_groups=len({r['group'] for r in rr if r['expected']==1}),
        answers=sum(r['status']=='assessed' for r in rr),
        wrong_answers=sum(r['status']=='assessed' and r['value']!=bool(r['expected']) for r in rr),
        positive_answers=sum(r.get('value') is True for r in rr),
        correct_positive_groups=len({r['group'] for r in rr if r.get('value') is True and r['expected']==1}),rows=rr)
        for a,rr in reports.items()}

def main():
    torch.set_num_threads(5)
    data = json.loads((OUT / 'dataset_manifest.json').read_bytes())
    full,crop = vectors()
    all_rows = data['train']+data['family_only_train']+data['test']+data['winter_test']+data['fresh_test']
    rgb = {r['key']:np.concatenate((full[r['key']],crop[r['key']])) for r in all_rows}
    z = np.load(OUT / 'gray_features.npz')
    gg = dict(zip(z['keys'],z['values'],strict=True))
    gray = {r['key']:np.concatenate((gg['full/'+r['key']],gg['crop/'+r['key']]))
        for r in data['train']+data['family_only_train']}
    baseline = json.loads((OUT / 'baseline/hierarchy_bundle.json').read_bytes())
    candidate = copy.deepcopy(baseline)
    training = {}
    families = [dict(r,model=family_of(r['model'])) for r in data['train']]
    families += [dict(r,model=r['family']) for r in data['family_only_train']]
    specialists = [r for r in data['train'] if r['model'] in MTZ_SPECIALIST]
    for name,rows in (('family',families),('mtz_specialist',specialists)):
        head,summary = recognition_train(rows,rgb,gray)
        candidate['heads'][name] = head
        candidate['calibration'][name] = copy.deepcopy(summary['thresholds'])
        if name == 'family':
            for value in candidate['calibration'][name].values():
                if value['threshold'] is not None:
                    value['threshold'] = max(.90,value['threshold'])
        training[name] = summary
        print('selected',name,summary['c'],summary['grayscale_training'],flush=True)
    import hashlib
    candidate['manifest_sha256'] = hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest()
    candidate['training_protocol'] = 'weather_round2_group_oof_rgb_or_gray_train_only'
    candidate['specialist_training'] = [{k:r[k] for k in ('key','group','model','sha256')} for r in specialists]
    comparisons = {name:{split:benchmark(bundle,data[split],rgb)
        for split in ('test','winter_test','fresh_test')} for name,bundle in (('baseline',baseline),('candidate',candidate))}
    save('hierarchy_candidate.json',candidate)
    save('recognition_training.json',training)
    save('recognition_comparison.json',comparisons)
    print('recognition',json.dumps({n:{s:{k:v[k] for k in ('total','family_correct','answers','wrong_answers')} for s,v in ss.items()} for n,ss in comparisons.items()}),flush=True)
    baseline_conditions = json.loads((OUT/'baseline/condition_bundle.json').read_bytes())
    conditions = copy.deepcopy(baseline_conditions)
    condition_training = {}
    scene_weather = {}
    for r in data['condition_rows']:
        with Image.open(ROOT / r['path']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
        scene_weather[r['key']] = np.concatenate((full[r['key']],weather_measurements(image)*5))
    for a in ('dirty','snow','rain','wet'):
        rows = [dict(r,model=str(r['condition'][a])) for r in data['condition_rows']
            if r['role']=='train' and r['condition'].get(a) is not None]
        held = [r for r in data['condition_rows'] if r['role']=='test' and r['condition'].get(a) is not None]
        assert not {r['group'] for r in rows} & {r['group'] for r in held}, a
        counts = Counter(r['model'] for r in rows)
        groups = {label:len({r['group'] for r in rows if r['model']==label}) for label in ('0','1')}
        if min(groups.values())<3:
            condition_training[a] = dict(skipped=True,groups=groups)
            continue
        options = {'crop':crop,'scene':full} if a=='dirty' else {'scene_weather':scene_weather} if a=='snow' else {'scene':full}
        candidates = {source:train(rows,v,min_groups=6 if a in ('dirty','snow') else 3) for source,v in options.items()}
        source = max(candidates,key=lambda k:candidates[k][1]['grouped_oof_balanced_accuracy'])
        head,summary = candidates[source]
        conditions['heads'][a] = head
        conditions['calibration'][a] = summary['thresholds']
        conditions.setdefault('feature_sources',{})[a] = source
        conditions.setdefault('training',{})[a] = summary
        condition_training[a] = dict(source=source,groups=groups,summary=summary,
            candidates={k:v[1]['grouped_oof_balanced_accuracy'] for k,v in candidates.items()})
        print('condition',a,source,groups,summary['grouped_oof_balanced_accuracy'],flush=True)
    held = [r for r in data['condition_rows'] if r['role']=='test']
    cm = {name:condition_benchmark(bundle,held,full,crop) for name,bundle in (('baseline',baseline_conditions),('candidate',conditions))}
    save('condition_candidate.json',conditions)
    save('condition_training.json',condition_training)
    save('condition_comparison.json',cm)
    print('conditions',json.dumps({n:{a:{k:v[k] for k in ('total','answers','wrong_answers','positive_answers','correct_positive_groups')} for a,v in aa.items()} for n,aa in cm.items()}),flush=True)

if __name__=='__main__':
    main()
