"""Train the SC-GS-style V4 residual field while retaining the complete V3 base."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.deformation.train_v3_auxiliary_mlp import main


if __name__ == '__main__':
    main(field_kind='scgs-v4')
