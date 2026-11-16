"""Publish only the guarded round-3 artifacts after diagnostics and source isolation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/robustness_round3';DEST=ROOT/'artifacts/mvp'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    gates=json.loads((OUT/'recognition_gates.json').read_bytes())
    assert all(gates['combined'].values()),gates['combined']
    checks=json.loads((OUT/'candidate_pipeline_verification.json').read_bytes())
    cases={r['case']:r for r in checks}
    assert len(cases['multiple']['objects'])==3
    assert len(cases['attachment']['objects'])==1 and cases['attachment']['ignored']==1
    assert cases['mvp_m004']['objects'][0]['image_quality']['low_light']
    assert cases['mvp_m004']['objects'][0]['condition']=='not_assessed'
    assert cases['mud_conflict']['objects'][0]['condition']=='dirty'
    partial=json.loads((OUT/'partial_training_manifest.json').read_bytes())
    assert not {r['group'] for r in partial['train']} & {r['group'] for r in partial['test']}
    assert all(r['role']=='feedback_train' for r in partial['feedback_regression'])
    data=json.loads((OUT/'dataset_manifest.json').read_bytes())
    training=data['train']+data['family_only_train']
    held=[r for split in ('test','winter_test','fresh_test','new_photos_test') for r in data[split]]
    for field in ('group','sha256'):
        assert not {r[field] for r in training} & {r[field] for r in held}
    for r in training+held+data['condition_rows']:
        assert digest(ROOT/r['path'])==r['sha256'],r['key']
    # Detect concurrent artifact changes before any replacement.
    for filename in ('hierarchy_bundle.json','region_bundle.json','condition_bundle.json'):
        assert digest(DEST/filename)==digest(OUT/'baseline'/filename),filename
    metrics=json.loads((OUT/'recognition_comparison.json').read_bytes())
    def normalise(value):
        return dict(total=value['total'],family_correct=value['family_correct'],family_answers=value['answers'],wrong_answers=value['wrong_answers'],
            rows=[dict(r,family_candidate=r['candidate']) for r in value['rows']])
    recognition=dict(baseline=normalise(metrics['baseline']['test']),selected=normalise(metrics['combined']['test']),
        new_control_baseline=normalise(metrics['baseline']['winter_test']),new_control_selected=normalise(metrics['combined']['winter_test']),
        round3_comparison=metrics,note='Frozen single-object features; full detection/OCR checks and photometric refusals are reported separately. Repeatedly examined diagnostic controls are not blinded validation.')
    region_metrics=json.loads((OUT/'baseline/region_control.json').read_bytes())
    region_metrics['round3_partial_comparison']=json.loads((OUT/'partial_ab_comparison.json').read_bytes())
    region_metrics['note']='The user partial-tractor case is training feedback. Independent region control uses 30 cropped real-machine views and one implement crop; negative diversity is limited.'
    outputs={'hierarchy_bundle.json':json.loads((OUT/'combined_candidate.json').read_bytes()),
        'region_bundle.json':json.loads((OUT/'partial_region_candidate.json').read_bytes()),
        'recognition_metrics.json':recognition,'region_control.json':region_metrics}
    for filename,obj in outputs.items():
        path=DEST/filename;temp=path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    manifest=dict(dataset_sha256=digest(OUT/'dataset_manifest.json'),
        deployed='known-family update + calibrated positive T5000 specialist + edge-fragment specialist + low-light refusal',
        unchanged=['type head','MTZ specialist','condition bundle'],
        artifacts={p.name:digest(p) for p in DEST.glob('*.json')},
        sources={p:digest(ROOT/p) for p in ('src/models/hierarchy.py','src/models/mvp_inference.py','src/models/image_visibility.py','scripts/predict_machinery_hierarchy.py')})
    (OUT/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Released guarded recognition and partial verifier; original type, MTZ specialist and condition heads retained.',flush=True)

if __name__=='__main__':main()
