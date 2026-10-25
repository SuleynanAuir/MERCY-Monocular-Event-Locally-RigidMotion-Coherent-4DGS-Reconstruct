"""Compare the V3 material finish with itself disabled; differences are not accuracy scores."""
import argparse
import io
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
from utils.gaussian_pair_renderer import GaussianPairRenderer
from utils.viewer_cache import EncodedFrameCache
from utils.viewer_state import validate_request


def pixels(packet):
    length = struct.unpack('<I', packet[:4])[0]
    return np.asarray(Image.open(io.BytesIO(packet[4+length:])).convert('RGB'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    renderer = GaussianPairRenderer(args.manifest)
    renderer.load()
    normals = renderer.detail_normals
    if normals is None:
        raise ValueError('Expected an enabled V3 detail finish')
    views = [(0, 0), (35, 20), (80, 0), (180, 0), (0, 35), (0, -35)]
    views += [(yaw, pitch) for yaw in (45, 135, 225, 270, 315) for pitch in (-25, 25)]
    rows = []
    for frame in (0, 25, 49):
        for view, (yaw, pitch) in enumerate(views):
            for mode in ('gaussian', 'clay', 'pseudocolor'):
                request = validate_request(dict(frame=frame, width=1024, height=1024, yaw=yaw,
                    pitch=pitch, fovy=renderer.info['fovy'], background='light', quality=95,
                    render_mode=mode), renderer.info['frame_count'])
                pair = []
                for enabled in (False, True):
                    renderer.detail_normals = normals if enabled else None
                    renderer.frames = EncodedFrameCache(max_bytes=1024**2, max_entries=1)
                    pair.append(pixels(renderer.render(request)))
                before, after = pair
                left_equal = np.array_equal(before[:, :1024], after[:, :1024])
                if not left_equal:
                    raise ValueError('Detail finish changed the V1 control panel')
                difference = np.abs(after[:, 1024:].astype(float)-before[:, 1024:].astype(float))
                foreground = before[:, 1024:].min(2) < 245
                rows.append(dict(frame=frame, view=view, yaw=yaw, pitch=pitch, mode=mode,
                    v1_unchanged=left_equal, v3_foreground_mean_difference=float(difference[foreground].mean())))
                Image.fromarray(after).save(args.output/f'{mode}-f{frame:03d}-v{view:02d}.jpg', quality=95)
                if view in (0, 1, 3, 5):
                    Image.fromarray(np.concatenate([before, after[:, 1024:]], 1)).save(
                        args.output/f'comparison-{mode}-f{frame:03d}-v{view:02d}.jpg', quality=95)
            print('validated', frame, view, flush=True)
    report = dict(note=__doc__, camera_pose_samples=48, display_modes=3, measurements=rows,
                  v1_unchanged=all(x['v1_unchanged'] for x in rows))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
