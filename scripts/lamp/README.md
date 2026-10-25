# Lamp viewer

Render `data/lamp_data` with the existing light UI, six-view preparation, per-view
detail boxes and synchronized RGB/event references. The two supplied models are
shown as V1 (`lamp.ply`) and V3 (`lamb_corrected_g0.ply`). These are supplied
appearance versions, not a newly trained or event-validated V3 reconstruction.

## Prepare

Use the project CUDA environment and Blender 5.2. Read the blend with embedded
Python execution disabled. Its existing render sampling is frames 1, 11, …, 491:
50 output frames at 30 FPS, matching both reference videos.

```bash
source scripts/setup/autodl_env.sh
LAMP_RUN=/root/autodl-tmp/scgs-results/lamp-20261025
blender -b --disable-autoexec data/lamp_data/lamp.blend \
  --python scripts/lamp/export_lamp_motion.py -- --output "$LAMP_RUN/parts.npz"
python scripts/lamp/prepare_lamp_viewer.py --data data/lamp_data \
  --parts "$LAMP_RUN/parts.npz" --output "$LAMP_RUN/viewer"
python scripts/viewer/prepare_reference_videos.py \
  --rgb data/lamp_data/lamp.mkv --events data/lamp_data/lamp_events.mp4 \
  --output "$LAMP_RUN/references" --white-background
```

All output destinations must be new. Source models and the blend are unchanged.
Add the following scene to an existing catalog, preserving its other entries, or
use this as a standalone `scenes.json` beside `viewer/` and `references/`:

```json
{
  "default": "lamp",
  "scenes": [{
    "id": "lamp", "name": "Lamp · 台灯", "type": "gaussian_pair",
    "path": "viewer/manifest.json", "cache": "cpu",
    "reference_path": "references/reference.json"
  }]
}
```

```bash
python viewer.py --scenes_path "$LAMP_RUN/scenes.json" --host 127.0.0.1 --port 8892
```

Forward the port over SSH, then open `http://127.0.0.1:8892/?scene=lamp`.
Available modes are native GS, soft white GS shading and pseudocolor. This scene
does not claim a reconstructed Mesh or Mesh+GS asset; those choices are disabled.

## Motion and checks

The exporter verifies unchanged local mesh geometry, fixed camera and topology
throughout the source animation. Camera-initialized geometric registration maps
Blender parts to GS; both versions use the same registration and part transforms.
Each Gaussian follows its nearest source part, retaining its supplied appearance.
Offset shade rows also follow their part: freezing them would leave a detached
cap when the lamp tilts. Registration and maximum-distance guards reject gross
misalignment; binding distances and extrapolated rows are recorded for review.
Nearest-surface assignment remains an approximation, especially at close joints.

Sequences validate canonical-state and frame checksums. The comparison renderer
requires identical cameras, frame counts and timestamps, and verifies both file
checksums. No RGB or event optimization is performed. RGB/event alignment is
checked independently before the reference manifest is accepted; the initial
event frame is empty because it has no preceding interval.

```bash
python -m unittest tests.viewer.test_gaussian_pair \
  tests.viewer.test_viewer_server tests.viewer.test_viewer_reference
```

## Refine the supplied V3 surface

The supplied corrected model has many low-opacity, displaced surface samples.
Increasing their count alone amplifies the ragged shade rim and grain on thin
rods. `refine_lamp_surface.py` instead uses the original GS as an explicit
geometry prior. It transfers bounded, reliable V3 appearance residuals only
within the same rigid part, gated by distance, normal, opacity and scale.
Deterministic tangent subdivision preserves parent position/covariance moments
and approximate optical-depth mass; children inherit their parent's part.

This is a display-oriented reconstruction using existing geometry, **not** new
observed detail, event fitting, or evidence of improved reconstruction accuracy.
The V1 sequence stays unchanged. V3 uses twice the render width and height,
then resolves to the requested display size before soft-shading evaluation.
The original shared material chart is retained so updating V3 does not recolor
V1 in clay or pseudocolor modes.

```bash
LAMP_QUALITY=/root/autodl-tmp/scgs-results/lamp-quality-20261025
LAMP_BASE="$LAMP_RUN/viewer-bound" # directory of the active baseline manifest
python scripts/lamp/refine_lamp_surface.py --data data/lamp_data \
  --parts "$LAMP_RUN/parts.npz" --binding-report "$LAMP_BASE/binding-report.json" \
  --output "$LAMP_QUALITY/candidate" --split 4
python scripts/lamp/evaluate_lamp_surface.py --data data/lamp_data \
  --parts "$LAMP_RUN/parts.npz" --binding-report "$LAMP_BASE/binding-report.json" \
  --candidate "$LAMP_QUALITY/candidate" --output "$LAMP_QUALITY/full-dynamic" \
  --frames 0,25,49 --resolution 1024 --full-orbit
python scripts/lamp/package_lamp_surface.py --baseline "$LAMP_BASE/manifest.json" \
  --candidate "$LAMP_QUALITY/candidate" --parts "$LAMP_RUN/parts.npz" \
  --binding-report "$LAMP_BASE/binding-report.json" --output "$LAMP_QUALITY/final"
python -m unittest tests.viewer.test_lamp_surface tests.viewer.test_gaussian_pair
```

Review the triptychs (V1 / supplied V3 / refined V3), then update only the Lamp
catalog entry to `final/manifest.json` and restart the viewer. The original
baseline files remain available for rollback. The evaluator covers six UI
presets plus ten oblique views at the start, middle and end of motion, in GS,
clay and pseudocolor. Its interior Laplacian and detached-pixel measurements
are appearance diagnostics, not ground-truth fidelity or event metrics.

The 2026-10-25 candidate contains 1,706,788 Gaussians (853,403 in the supplied
V3), with 674,300 following moving parts. Across the 48 matched camera/pose
samples, interior high-frequency roughness fell by 60.40% in GS, 53.91% in clay,
and 50.48% in pseudocolor. Small detached silhouette components fell from 707
pixels total to zero at 1024 px; minimum silhouette IoU against the original
geometry prior was 0.9830. These measurements are resolution-dependent and
cannot distinguish every real detail from noise. Matched live-render checks
also confirmed byte-identical V1 panels in all three modes after deployment.

## Make V3 surface detail more readable

The geometry-prior reconstruction above intentionally resembles the original
model: increasing sampling does not invent new features. For a stronger visual
distinction, `prepare_lamp_detail.py` adds an explicitly declared **display
finish** to V3. It transfers smooth normals from the supplied Blender surface
within each rigid part, rejects poorly aligned directions, and blends distant
matches back toward the Gaussian normal. Directions are stored in each
Gaussian's local rotation frame and follow the existing animation exactly.
At render time, agreement with normals derived from the visible Gaussian depth
also gates the finish, so an inconsistent source normal cannot add a false crease.
The static base retains its original shading. A smooth, canonical-height envelope
fades the enhancement in above the base toward the lower joint; it follows the
object rather than creating a camera-dependent mask or a hard lighting boundary.

The renderer uses the extra normal pass for soft highlights, local contrast and
small-radius contact shading. V1 pixels, V3 positions, silhouettes, point count,
camera and motion are unchanged. Lighting is a camera-relative display finish,
not a calibrated shared-light reconstruction experiment; brighter highlights or
more visible grooves are **not evidence of higher training/reconstruction
accuracy**. Both the UI labels and scene description declare this distinction.

```bash
LAMP_DETAIL=/root/autodl-tmp/scgs-results/lamp-detail-20261025
python scripts/lamp/prepare_lamp_detail.py \
  --manifest "$LAMP_QUALITY/final/manifest.json" \
  --parts "$LAMP_RUN/parts.npz" --binding-report "$LAMP_BASE/binding-report.json" \
  --blend data/lamp_data/lamp.blend --blender /absolute/path/to/blender \
  --output "$LAMP_DETAIL/final"
python scripts/lamp/evaluate_lamp_detail.py \
  --manifest "$LAMP_DETAIL/final/manifest.json" --output "$LAMP_DETAIL/validation"
python -m unittest tests.viewer.test_lamp_detail tests.viewer.test_lamp_surface \
  tests.viewer.test_gaussian_pair
```

The evaluator renders the finish enabled/disabled at 16 angles, three source
poses and three display modes; every comparison asserts an unchanged V1 panel.
Its pixel-difference measurement quantifies visibility of the change, not image
quality. Review the images before selecting this manifest in the scene catalog.
The previous geometry-only manifest remains available as a rollback and for
comparisons without the display finish.
