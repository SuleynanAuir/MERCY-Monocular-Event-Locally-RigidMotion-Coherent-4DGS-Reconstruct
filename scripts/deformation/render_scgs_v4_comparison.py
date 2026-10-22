"""Render V3 versus the SC-GS-style V4 at matching original times and cameras."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.deformation.render_v3_auxiliary_comparison import main


if __name__ == '__main__':
    main(field_kind='scgs-v4')
