import hashlib,json
from scripts.check_mvp_release import check_release

def test_release_check_finds_changed_weights(tmp_path):
    folder=tmp_path/'artifacts/mvp';folder.mkdir(parents=True)
    artifact=folder/'demo.bin';artifact.write_bytes(b'weights');source=tmp_path/'demo.py';source.write_text('pass',encoding='utf8')
    manifest=tmp_path/'release.json';manifest.write_text(json.dumps({'round':'test','artifacts':{'demo.bin':hashlib.sha256(artifact.read_bytes()).hexdigest()},'sources':{'demo.py':hashlib.sha256(source.read_bytes()).hexdigest()}}),encoding='utf8')
    assert check_release(manifest,tmp_path)['passed']
    artifact.write_bytes(b'changed')
    result=check_release(manifest,tmp_path)
    assert not result['passed'] and 'demo.bin' in result['errors'][0]

def test_release_check_rejects_outside_paths(tmp_path):
    manifest=tmp_path/'release.json';manifest.write_text(json.dumps({'artifacts':{'../../../outside.bin':'bad'},'sources':{'demo.py':'bad'}}),encoding='utf8')
    assert any('outside project' in e for e in check_release(manifest,tmp_path)['errors'])
