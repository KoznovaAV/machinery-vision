"""Confirm expansion under the incumbent fixed-C protocol and record deployment gates."""
import copy
import hashlib
import json
import numpy as np
from scripts.train_winter_round2 import ROOT,OUT,vectors,benchmark,condition_benchmark,save
from scripts.train_mvp import train
from src.config.machinery_taxonomy import family_of

def main():
    data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    full,crop=vectors()
    all_rows=data['train']+data['family_only_train']+data['test']+data['winter_test']+data['fresh_test']
    rgb={r['key']:np.concatenate((full[r['key']],crop[r['key']])) for r in all_rows}
    baseline=json.loads((OUT/'baseline/hierarchy_bundle.json').read_bytes())
    fixed=copy.deepcopy(baseline)
    rows=[dict(r,model=family_of(r['model'])) for r in data['train']]
    rows += [dict(r,model=r['family']) for r in data['family_only_train']]
    head,summary=train(rows,rgb,tune=False,min_groups=10)
    fixed['heads']['family']=head
    fixed['calibration']['family']=copy.deepcopy(summary['thresholds'])
    for value in fixed['calibration']['family'].values():
        if value['threshold'] is not None:
            value['threshold']=max(.90,value['threshold'])
    fixed['manifest_sha256']=hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest()
    fixed['training_protocol']='weather_round2_expanded_family_original_specialist_fixed_c'
    save('hierarchy_fixed_c.json',fixed)
    save('recognition_fixed_c_training.json',summary)
    metrics=json.loads((OUT/'recognition_comparison.json').read_bytes())
    candidate=json.loads((OUT/'hierarchy_candidate.json').read_bytes())
    for value in candidate['calibration']['family'].values():
        if value['threshold'] is not None:
            value['threshold']=max(.90,value['threshold'])
    save('hierarchy_candidate.json',candidate)
    metrics['candidate']={s:benchmark(candidate,data[s],rgb) for s in ('test','winter_test','fresh_test')}
    metrics['fixed_c']={s:benchmark(fixed,data[s],rgb) for s in ('test','winter_test','fresh_test')}
    save('recognition_comparison.json',metrics)
    print('fixed_c',json.dumps({s:{k:v[k] for k in ('total','family_correct','answers','wrong_answers','label_correct')} for s,v in metrics['fixed_c'].items()}),flush=True)
    old=metrics['baseline']
    gates={}
    for name in ('candidate','fixed_c'):
        new=metrics[name]
        gates[name]=dict(
            old_family_non_regression=new['test']['family_correct']>=old['test']['family_correct'],
            old_wrong_non_regression=new['test']['wrong_answers']<=old['test']['wrong_answers'],
            old_labels_non_regression=new['test']['label_correct']>=old['test']['label_correct'],
            winter_improvement=new['winter_test']['family_correct']>old['winter_test']['family_correct'],
            winter_wrong_non_regression=new['winter_test']['wrong_answers']<=old['winter_test']['wrong_answers'],
            fresh_improvement=new['fresh_test']['family_correct']>=old['fresh_test']['family_correct'],
            fresh_wrong_non_regression=new['fresh_test']['wrong_answers']<=old['fresh_test']['wrong_answers'])
    baseline_conditions=json.loads((OUT/'baseline/condition_bundle.json').read_bytes())
    conditions=copy.deepcopy(baseline_conditions)
    candidate_conditions=json.loads((OUT/'condition_candidate.json').read_bytes())
    # Recognition of visible dirt is positive evidence. Absence is not proof of cleanliness.
    for key in ('heads','calibration','feature_sources','training'):
        conditions.setdefault(key,{})['dirty']=copy.deepcopy(candidate_conditions[key]['dirty'])
    conditions['calibration']['dirty']['0']=dict(threshold=None,groups=0,reason='Absence of detected dirt is not evidence of cleanliness; positive-evidence MVP policy')
    conditions['training_manifest_sha256']=fixed['manifest_sha256']
    held=[r for r in data['condition_rows'] if r['role']=='test']
    selected_conditions=condition_benchmark(conditions,held,full,crop)
    cm=json.loads((OUT/'condition_comparison.json').read_bytes())
    cm['positive_evidence']=selected_conditions
    save('condition_comparison.json',cm)
    save('condition_positive_evidence.json',conditions)
    dirty_old=cm['baseline']['dirty']; dirty_new=selected_conditions['dirty']
    condition_gate=(dirty_new['wrong_answers']<=dirty_old['wrong_answers'] and dirty_new['answers']>dirty_old['answers'])
    save('deployment_gates.json',dict(recognition=gates,condition_positive_evidence=condition_gate,
        reason='Frozen controls are release gates, never parameter selection. Neither rain nor wet is promoted without positive independent validation.'))
    print('gates',json.dumps(gates),flush=True)
    print('dirty_positive_evidence',json.dumps({k:v for k,v in dirty_new.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    main()
