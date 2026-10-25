# Metronome and Microphone viewer

The local microphone directory is spelled `data/micphone_data`; its UI scene ID
is `microphone`. Each dataset supplies an original PLY (V1), `corrected_g0.ply`
(V3), Blender animation, camera intrinsics, RGB video and event visualization.
These are supplied versions, not newly event-trained results.

Use a CUDA environment with the project's Gaussian rasterizer and Blender 5.2.
Always read the blend with `--disable-autoexec`; source files are never saved.
Choose a new output directory for each preparation run.

```bash
RUN=/root/autodl-tmp/scgs-results/objects-20261025
blender -b --disable-autoexec data/metronome_data/metronome.blend \
  --python scripts/lamp/export_lamp_motion.py -- \
  --name Metronome --output "$RUN/metronome-parts.npz"
python scripts/lamp/prepare_object_viewer.py --data data/metronome_data \
  --stem metronome --name 'Metronome · 节拍器' \
  --parts "$RUN/metronome-parts.npz" --output "$RUN/metronome"
blender -b --disable-autoexec data/micphone_data/micphone.blend \
  --python scripts/lamp/export_lamp_motion.py -- \
  --name Microphone --output "$RUN/microphone-parts.npz"
python scripts/lamp/prepare_object_viewer.py --data data/micphone_data \
  --stem micphone --name 'Microphone · 麦克风' \
  --parts "$RUN/microphone-parts.npz" --output "$RUN/microphone"
python scripts/viewer/prepare_reference_videos.py \
  --rgb data/metronome_data/metronome.mkv \
  --events data/metronome_data/metronome_events.mp4 \
  --output "$RUN/metronome-white-reference" --white-background --rgb-background 69 69 69
python scripts/viewer/prepare_reference_videos.py \
  --rgb data/micphone_data/micphone.mkv \
  --events data/micphone_data/micphone_event.mp4 \
  --output "$RUN/microphone-reference" --white-background
```

The exporter validates topology, local geometry, fixed camera and the complete
source timeline. Registration uses the supplied GS camera when present; otherwise
it searches proper PCA orientations and trimmed similarity transforms. Both GS
versions share exactly the same camera and rigid-part transforms. Review the
result against RGB at multiple motion phases before selecting it in the catalog;
geometry registration alone cannot disambiguate perfectly symmetric objects.
Attachment is an approximation and is rejected if surface distances are excessive.

Add these entries to `scenes.json`, preserving any existing scenes:

```json
{
  "default": "metronome",
  "scenes": [
    {"id":"metronome", "name":"Metronome · 节拍器", "type":"gaussian_pair",
     "path":"metronome/manifest.json", "cache":"cpu",
     "reference_path":"metronome-white-reference/reference.json"},
    {"id":"microphone", "name":"Microphone · 麦克风", "type":"gaussian_pair",
     "path":"microphone/manifest.json", "cache":"cpu",
     "reference_path":"microphone-reference/reference.json"}
  ]
}
```

```bash
python viewer.py --scenes_path "$RUN/scenes.json" --host 127.0.0.1 --port 8893
python -m unittest tests.viewer.test_object_preparation tests.viewer.test_gaussian_pair \
  tests.viewer.test_reference_matte tests.viewer.test_viewer_reference
```

Forward port 8893 over SSH. Use `/?scene=metronome` or `/?scene=microphone`.
Available paired display modes are GS, soft white shading and pseudocolor; these
assets do not provide an extracted Mesh or Mesh+GS scene. Six-view precaching,
per-view detail regions, synchronized references and full-video export use the
existing viewer. A constant RGB backdrop override changes only border-connected
background pixels, validates the border color and preserves enclosed dark parts.
This is display-only; original RGB and event files remain unchanged.

## Surface-constrained V3 refinement

For objects whose corrected GS contains duplicate points, rough contours or a
moving part's silhouette baked onto the static body, the supplied Blender
surface can serve as an explicit geometry prior. This is **not** a new
reconstruction from events: it re-samples the known source surface as tangent
Gaussians and estimates robust color per material/planar patch from the supplied
V3. It can suppress texture detail within a patch; inspect textured objects
before enabling it. It should not be reported as an event-training metric gain.

```bash
QUALITY=/root/autodl-tmp/scgs-results/metronome-quality-20261026
blender -b --disable-autoexec data/metronome_data/metronome.blend \
  --python scripts/lamp/export_object_surface.py -- \
  --parts "$RUN/metronome-parts.npz" \
  --binding-report "$RUN/metronome/binding-report.json" \
  --output "$QUALITY/source-surface.npz"
python scripts/lamp/refine_object_surface.py \
  --surface "$QUALITY/source-surface.npz" \
  --baseline "$RUN/metronome/manifest.json" \
  --parts "$RUN/metronome-parts.npz" \
  --binding-report "$RUN/metronome/binding-report.json" \
  --output "$QUALITY/candidate" --spacing .0015
python -m unittest tests.viewer.test_object_surface tests.viewer.test_gaussian_pair \
  tests.viewer.test_lamp_detail
```

Every new Gaussian inherits its source triangle's rigid part. Original motion,
frame timestamps, camera and V1 assets are retained. Source identity, bindings
and normal assets are protected by checksums. The original shared material chart
is retained so V1 does not change when V3 geometry changes. V3 uses 2x rendering
and downsampling to reduce contour aliasing.

`manifest.json` enables the new geometry and patch appearance;
`manifest-detail.json` also enables transported surface-normal lighting,
contact shading and soft highlights. Both remain Gaussian rendering, not a
replacement RGB video. The latter is a display finish, not additional geometry
or training. Validate all six views and multiple swing phases before pointing
the metronome catalog entry at either manifest; retain the previous catalog and
all baseline assets for rollback. Existing RGB/event references remain unchanged.
