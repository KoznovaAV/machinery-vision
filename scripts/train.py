"""Train a separate candidate: python -m scripts.train --task conditions."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from src.config.runtime import ROOT
from src.data.manifest import audit


def fingerprints():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "artifacts/mvp").iterdir()
        if p.is_file()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", choices=("families", "conditions", "regions", "mtz"), required=True
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = (
        a.output
        or ROOT
        / "runs/training"
        / (a.task + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    ).resolve()
    if not out.is_relative_to(ROOT.resolve()) or out.exists():
        p.error("Use a new output directory inside the project")
    report = audit()
    if not report["passed"]:
        raise ValueError(report["errors"])
    out.mkdir(parents=True)
    before = fingerprints()
    import torch

    torch.set_num_threads(5)
    from src.training import candidates

    candidates.OUT = out
    try:
        getattr(candidates, "train_" + a.task)()
    finally:
        if before != fingerprints():
            raise RuntimeError("Installed artifacts unexpectedly changed")
    (out / "run.json").write_text(
        json.dumps(
            {
                "task": a.task,
                "installed_artifacts_unchanged": True,
                "automatic_deployment": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print("Candidate only:", out)


if __name__ == "__main__":
    main()
