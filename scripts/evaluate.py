"""Compare complete inference against the frozen 20-scene diagnostic control."""

import argparse, json
from datetime import datetime
from pathlib import Path
from src.config.runtime import ROOT
from src.data.manifest import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    out = (
        a.output or ROOT / "runs/evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        p.error("Output must stay in the project")
    out.mkdir(parents=True, exist_ok=True)
    from scripts.predict import run

    reports = []
    for i, c in enumerate(read(ROOT / "datasets/scenes.json")[: a.limit]):
        result = run(
            ROOT / c["path"],
            ROOT / "artifacts/mvp/hierarchy_bundle.json",
            out / str(i) / "prediction.json",
        )
        objects = result.get("objects", [])
        checks = {
            "object_count": len(objects) == c["expected_objects"],
            "uncertain_count": len(result.get("uncertain_regions", [])) == c["expected_uncertain"],
            "ignored_reasons": [r["reason"] for r in result.get("ignored_regions", [])]
            == c["expected_ignored"],
        }
        for j, expected in enumerate(c["expected_predictions"]):
            checks["prediction_" + str(j)] = j < len(objects) and all(
                objects[j].get(k) == v for k, v in expected.items()
            )
        reports.append({"name": c["name"], "passed": all(checks.values()), "checks": checks})
        print(c["name"], reports[-1]["passed"], flush=True)
    summary = {
        "passed": all(r["passed"] for r in reports),
        "cases": len(reports),
        "rows": reports,
        "note": "Repeated diagnostic control, not a blinded accuracy estimate.",
    }
    (out / "checks.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    raise SystemExit(not summary["passed"])


if __name__ == "__main__":
    main()
