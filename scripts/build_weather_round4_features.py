"""Use the established frozen feature pipeline for round-4 RGB views."""
from pathlib import Path
import scripts.build_robustness_features as builder
if __name__=='__main__':
    builder.OUT=Path(__file__).resolve().parents[1]/'output/weather_round4'
    builder.b.CROP_DIRS=('output/weather_round4/views',)+builder.b.CROP_DIRS
    builder.main()
