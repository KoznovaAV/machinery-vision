"""Freeze reviewed weather additions while preserving all earlier controls."""

import hashlib
import json
from pathlib import Path
from collections import Counter

from scripts.audit_round4_candidates import canonical_source, fingerprint, distance
from src.config.machinery_taxonomy import family_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/winter_round2"


def main():
    old = json.loads((ROOT / "output/mvp/dataset_manifest.json").read_bytes())
    controls = json.loads((OUT / "baseline/recognition_metrics.json").read_bytes())["new_control_selected"]["rows"]
    reviewed = {r["key"]:r for r in old["new_review"]}
    winter = [dict(reviewed[r["key"]],expected_family=r["expected_family"]) for r in controls]
    known = {}
    old_rows = old["train"] + old["family_only_train"] + old["test"] + old["condition_rows"] + winter
    for row in old_rows:
        if row["sha256"] not in known:
            known[row["sha256"]] = fingerprint(ROOT / row["path"])
    sources = {canonical_source(r["source"]) for r in old_rows
               if r.get("source") and str(r["source"]).startswith("http")}
    decisions = json.loads((OUT / "visual_decisions.json").read_bytes())
    rows = json.loads((OUT / "downloaded.json").read_bytes())
    audit = []
    accepted = []
    for row in rows:
        if row["status"] != "downloaded":
            continue
        decision = decisions[row["id"]]
        item = dict(row, **decision)
        item["key"] = "winter2/" + row["id"]
        item["source"] = canonical_source(row["source"])
        item["group"] = decision.get("group_override") or (
            "source-" + hashlib.sha256(item["source"].encode()).hexdigest()[:16])
        fp = fingerprint(ROOT / row["path"])
        assert fp["sha256"] == row["sha256"]
        if decision["decision"] == "accepted":
            nearest = min((distance(fp, old_fp), sha) for sha,old_fp in known.items())
            if row["sha256"] in known or item["source"] in sources or nearest[0] <= 6:
                item.update(decision="duplicate_or_protected",nearest_sha256=nearest[1])
            else:
                accepted.append(item)
                known[row["sha256"]] = fp
                sources.add(item["source"])
        audit.append(item)
    new_train = [r for r in accepted if r["split"] == "train"]
    fresh = [dict(r, expected_family=r.get("family", family_of(r.get("model"))))
             for r in accepted if r["split"] == "test" and (r.get("family") or r.get("model"))]
    state_rows = list(old["condition_rows"])
    # Author caption explicitly identifies this exposure as in rain.
    corrections = []
    for row in state_rows:
        if row["key"] == "mvp/m021":
            corrections.append(dict(key=row["key"],before=row["condition"],basis="Photographer caption 'Le parasol, pratique sous la pluie', Le Progres 2019-09-08."))
            row["condition"] = dict(row["condition"],rain=1)
    for row in accepted:
        if row.get("condition") is not None:
            state_rows.append(dict(row,role="test" if row["id"] == "b003" else row["split"]))
    manifest = dict(
        train=old["train"] + [r for r in new_train if r.get("model")],
        family_only_train=old["family_only_train"] + [r for r in new_train if r.get("family") and not r.get("model")],
        test=old["test"], winter_test=winter, fresh_test=fresh,
        condition_rows=state_rows, new_review=audit, label_corrections=corrections,
    )
    train = manifest["train"] + manifest["family_only_train"]
    held = manifest["test"] + winter + fresh
    assert not ({r["sha256"] for r in train} & {r["sha256"] for r in held})
    assert not ({r["group"] for r in train} & {r["group"] for r in held})
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("accepted",len(accepted),"models",Counter(r.get("model") or r.get("family") or "state_only" for r in accepted))
    print("recognition",len(manifest["train"]),len(manifest["family_only_train"]),len(manifest["test"]),len(winter),len(fresh))
    print("states",Counter(r["role"] for r in state_rows))
    print("positive groups",{k:len({r["group"] for r in state_rows if r["condition"].get(k)==1}) for k in ("dirty","snow","rain","wet")})


if __name__ == "__main__":
    main()
