"""Export canonical Gaussians + analytic deform.pth, without training."""

import argparse
import json
from pathlib import Path

from utils.rigid_checkpoint import RigidRotation, export_rigid_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--motion-summary", type=Path,
                        help="Reuse motion from a render_rotation.py summary.json instead of motion flags")
    parser.add_argument("--center", nargs=3, type=float)
    parser.add_argument("--axis", nargs=3, type=float)
    parser.add_argument("--rpm", type=float)
    parser.add_argument("--ramp-seconds", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--direction", choices=["cw", "ccw"])
    args = parser.parse_args()
    if args.motion_summary:
        if any(getattr(args, key) is not None for key in ("center", "axis", "rpm", "ramp_seconds", "duration", "direction")):
            parser.error("Use either --motion-summary or explicit motion flags, not both")
        data = json.loads(args.motion_summary.read_text())
        motion = RigidRotation(data["center"], data["unit_axis"], data["target_rpm"], data["ramp_seconds"],
                               data["duration_seconds"], data["direction_viewed_from_positive_axis"])
    else:
        if args.center is None:
            parser.error("--center is required unless --motion-summary is supplied")
        motion = RigidRotation(args.center, args.axis if args.axis is not None else [0, 0, 1],
                               args.rpm if args.rpm is not None else 60,
                               args.ramp_seconds if args.ramp_seconds is not None else 2,
                               args.duration if args.duration is not None else 6,
                               args.direction if args.direction is not None else "cw")
    payload = export_rigid_bundle(args.gs_path, args.output, motion)
    print(json.dumps(payload, indent=2))
    print(f"Exported: {args.output / 'point_cloud.ply'} and {args.output / 'deform.pth'}")


if __name__ == "__main__":
    main()
