"""Read-only integrity check of an explicitly versioned local MVP release."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check_release(manifest_path,root=ROOT):
    root=Path(root).resolve();manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'));errors=[]
    for section,prefix in (('artifacts','artifacts/mvp'),('sources','')):
        for name,expected in manifest.get(section,{}).items():
            target=(root/prefix/name).resolve()
            if not target.is_relative_to(root):errors.append(f'{section}: path outside project: {name}');continue
            if not target.is_file():errors.append(f'{section}: missing: {name}');continue
            if digest(target)!=expected:errors.append(f'{section}: changed: {name}')
    if not manifest.get('artifacts') or not manifest.get('sources'):errors.append('Missing artifact/source fingerprints')
    return dict(release=manifest.get('round'),passed=not errors,errors=errors,
                note='Integrity only; accuracy and full inference require the recorded validation checks.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);args=parser.parse_args()
    path=args.manifest
    if path is None:
        marker=ROOT/'artifacts/mvp/current_release.json'
        if not marker.is_file():parser.error('No current release marker; specify --manifest')
        value=json.loads(marker.read_bytes());path=(ROOT/value['manifest']).resolve()
        if not path.is_relative_to(ROOT.resolve()):parser.error('Manifest outside project')
    result=check_release(path);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
