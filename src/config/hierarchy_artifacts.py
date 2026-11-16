"""The current model has exactly one artifact directory."""

from src.config.runtime import ROOT


def hierarchy_bundle_path():
    return ROOT / "artifacts/mvp/hierarchy_bundle.json"
