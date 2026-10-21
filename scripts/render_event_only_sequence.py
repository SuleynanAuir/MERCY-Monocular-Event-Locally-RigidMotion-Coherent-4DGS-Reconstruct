#!/usr/bin/env python3
"""Export a custom event-only checkpoint and its genuinely posed G0 sequence.

Inputs are only an explicit G0 and checkpoint. Video is a generated output,
never supervision. Existing output directories/files are not overwritten.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sequence_times(last_time, fps):
    """Sample requested cadence and append the exact endpoint if off-grid."""
    if not math.isfinite(last_time) or last_time <= 0 or not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("Duration must be positive and FPS must be in (0, 120]")
    count = int(math.ceil(last_time * fps))
    if count > 10000:
        raise ValueError("Preview exceeds 10001 frames; request a lower FPS")
    values = [min(index / fps, last_time) for index in range(count + 1)]
    values[-1] = last_time
    return values


def requested_ply_times(value, last_time):
    if not value:
        return []
    try:
        result = sorted(set(float(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise ValueError("--write-ply-times must be comma-separated seconds") from error
    if len(result) > 8 or any(not math.isfinite(t) or t < 0 or t > last_time for t in result):
        raise ValueError("Export at most 8 PLY times, all inside the motion interval")
    return result


def _ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None
    return executable if Path(executable).is_file() else None


def _copy_new(source, destination):
    with Path(source).open("rb") as reader, Path(destination).open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def write_posed_ply(destination, g0, xyz, rotation):
    """Keep every source attribute/element; change only xyz and wxyz fields."""
    from plyfile import PlyData
    if tuple(xyz.shape) != (g0.count, 3) or tuple(rotation.shape) != (g0.count, 4):
        raise ValueError("Posed PLY rows must match the complete original G0")
    records = g0.raw_vertex.copy()
    positions = xyz.detach().cpu().numpy()
    quaternions = rotation.detach().cpu().numpy()
    for index, name in enumerate(("x", "y", "z")):
        records[name] = positions[:, index]
    for index in range(4):
        records[f"rot_{index}"] = quaternions[:, index]
    vertex = copy.copy(g0.ply["vertex"])
    vertex.data = records
    elements = [vertex if element.name == "vertex" else element for element in g0.ply.elements]
    result = PlyData(elements, text=g0.ply.text, byte_order=g0.ply.byte_order,
                     comments=list(g0.ply.comments), obj_info=list(g0.ply.obj_info))
    with Path(destination).open("xb") as output:
        result.write(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--write-ply-times", default="", help="Optional <=8 times such as 0,6.65; no per-frame PLY export")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.width <= 0 or args.width > 4096:
        raise ValueError("Width must be between 1 and 4096")
    import torch
    from PIL import Image
    from utils.event_only_checkpoint import load_event_only_checkpoint
    from utils.event_only_inputs import sha256_file
    from utils.event_only_renderer import EventOnlyRenderer, make_camera

    motion, camera, payload = load_event_only_checkpoint(args.checkpoint, args.g0)
    last_time = float(payload["times"][-1])
    timestamps = sequence_times(last_time, args.fps)
    ply_times = requested_ply_times(args.write_ply_times, last_time)
    render_camera = make_camera(camera, width=args.width)
    width, height = render_camera.width, render_camera.height
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None and width * height * len(timestamps) > 128 * 1024**2:
        raise ValueError("WebP fallback would exceed its memory bound; install ffmpeg or lower preview FPS/width")
    parent = args.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Reserve original/sparse PLYs, checkpoint, and a conservative raw-RGB video
    # upper bound plus 256 MiB. No automatic Gaussian decimation is performed.
    required_bytes = (args.g0.stat().st_size * (1 + len(ply_times)) + args.checkpoint.stat().st_size +
                      width * height * 3 * len(timestamps) + 256 * 1024**2)
    if shutil.disk_usage(parent).free < required_bytes:
        raise OSError(f"Insufficient disk space: require at least {required_bytes} bytes")
    args.output.mkdir()
    _copy_new(args.g0, args.output / "G0.ply")
    _copy_new(args.checkpoint, args.output / "deform.pth")
    if sha256_file(args.output / "G0.ply") != payload["input_sha256"]["g0"]:
        raise ValueError("G0 changed during export; bundle rejected")
    renderer = EventOnlyRenderer(args.output / "G0.ply", camera)
    key_indices = {min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - last_time * fraction))
                   for fraction in (0., .25, .5, .75, 1.)}
    keyframes, exported_plys, fallback_frames = [], [], []
    video_name = "preview.mp4" if ffmpeg else "preview.webp"
    process = None
    error_file = tempfile.TemporaryFile()
    if ffmpeg:
        process = subprocess.Popen(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-video_size", f"{width}x{height}", "-framerate", str(args.fps), "-i", "pipe:0",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(args.output / video_name)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file)
    try:
        with torch.no_grad():
            for index, timestamp in enumerate(timestamps):
                xyz, rotation = motion.deform(renderer.g0.xyz, renderer.g0.rotation, timestamp)
                rendered = renderer.render(xyz, rotation, width=args.width)
                pixels = (rendered["rgb"].clamp(0, 1).permute(1, 2, 0).mul(255).round()
                          .to(dtype=torch.uint8).cpu().numpy())
                image = Image.fromarray(pixels, mode="RGB")
                if index in key_indices:
                    name = f"keyframe-{index:05d}.png"
                    with (args.output / name).open("xb") as output:
                        image.save(output, format="PNG")
                    keyframes.append({"frame": index, "time_seconds": timestamp, "file": name})
                if process:
                    process.stdin.write(pixels.tobytes())
                else:
                    fallback_frames.append(image)
                if index % 25 == 0 or index + 1 == len(timestamps):
                    print(json.dumps({"rendered": index + 1, "frames": len(timestamps),
                                      "time_seconds": timestamp}), flush=True)
                del rendered, xyz, rotation
            for index, timestamp in enumerate(ply_times):
                xyz, rotation = motion.deform(renderer.g0.xyz, renderer.g0.rotation, timestamp)
                name = f"posed-{index:02d}-t{timestamp:.6f}.ply"
                write_posed_ply(args.output / name, renderer.g0, xyz, rotation)
                exported_plys.append({"time_seconds": timestamp, "file": name})
                del xyz, rotation
        if process:
            process.stdin.close()
            code = process.wait(timeout=60)
            if code:
                error_file.seek(0)
                raise RuntimeError(f"ffmpeg failed ({code}): {error_file.read().decode(errors='replace')}")
        else:
            with (args.output / video_name).open("xb") as output:
                fallback_frames[0].save(output, format="WEBP", save_all=True,
                                        append_images=fallback_frames[1:], duration=round(1000 / args.fps),
                                        loop=0, quality=95, method=4)
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        error_file.close()
    manifest = {
        "format": "event_only_sequence_preview_v1", "checkpoint_format": payload["format"],
        "source_gaussians": renderer.g0.count, "source_file": "G0.ply", "motion_file": "deform.pth",
        "preview_file": video_name, "frame_count": len(timestamps), "fps": args.fps,
        "sample_times_seconds": timestamps, "includes_exact_endpoint": True,
        "video_timing_note": "Constant-rate preview; the exact final source time may be off the regular sampling grid.",
        "render_width": width, "render_height": height,
        "video_width": width + (width % 2 if ffmpeg else 0),
        "video_height": height + (height % 2 if ffmpeg else 0),
        "keyframes": keyframes, "optional_posed_plys": exported_plys,
        "input_sha256": payload["input_sha256"], "training_summary": payload["training_summary"],
        "checkpoint_sha256": sha256_file(args.output / "deform.pth"),
        "appearance": "Original G0 SH, scales, opacity, and other properties remain unchanged; only xyz/quaternion move.",
        "original_scgs_network_weights": False, "approved_for_event_accuracy": False,
        "limitations": ["Video is a generated preview, never an input or validation reference.",
                        "Single-view events do not uniquely identify unseen 3D motion.",
                        "Export success alone does not establish event accuracy or physical correctness."],
    }
    with (args.output / "sequence.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "preview": video_name, "frames": len(timestamps),
                      "optional_ply_count": len(ply_times)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
