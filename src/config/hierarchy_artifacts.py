"""Locate a locally trained bundle or the checked-in evaluation snapshot."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def hierarchy_bundle_path() -> Path:
    mvp = ROOT / "artifacts/mvp/hierarchy_bundle.json"
    if mvp.is_file():
        return mvp
    local = ROOT / "output/continuation_round6/hierarchy_bundle.json"
    if local.is_file():
        return local
    return ROOT / "artifacts/hierarchy/hierarchy_bundle.json"
