from pathlib import Path
import scripts.build_robustness_features as b
if __name__=='__main__':
    b.OUT=Path(__file__).resolve().parents[1]/'output/trailer_round5';b.b.CROP_DIRS=('output/trailer_round5/views','output/weather_round4/views')+b.b.CROP_DIRS;b.main()
