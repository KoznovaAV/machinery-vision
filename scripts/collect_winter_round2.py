"""Collect a separate weather round while keeping earlier manifests intact."""

from pathlib import Path

import scripts.collect_mvp_photos as collection


if __name__ == "__main__":
    collection.OUT = Path(__file__).resolve().parents[1] / "output/winter_round2"
    collection.main()
