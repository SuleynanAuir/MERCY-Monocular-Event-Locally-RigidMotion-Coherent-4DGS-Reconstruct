"""Train V3's event-supervised local deformation with the audited MLP pipeline.

Legacy serialization and provenance keys remain unchanged for compatibility.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.deformation.train_v4_local_mlp import main


if __name__ == '__main__':
    main()
