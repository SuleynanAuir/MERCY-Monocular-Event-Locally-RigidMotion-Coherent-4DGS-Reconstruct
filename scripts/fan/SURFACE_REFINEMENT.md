# Fan V3 surface refinement

This pipeline refines the existing canonical Fan Gaussians. It does not train
from RGB frames or claim event-supervised reconstruction. The deployed source
has video-fitted rigid motion; complete raw Fan events were unavailable during
this run. Existing holes, occluded geometry and high-opacity detached artifacts
cannot be reliably reconstructed from these inputs alone.

## Method

- Fit a robust, bilateral moving-least-squares plane within each motion group.
  Normal agreement, color similarity and robust height trimming reject unrelated
  layers. Thin lines, boundary neighborhoods and ill-conditioned fits are protected.
- Apply bounded normal displacements, orientation alignment and local DC-color
  regularization. The selected run uses 256 neighbors and a 0.003 displacement
  bound in source coordinates. No frame-dependent optimization is introduced.
- Split 300,000 supported large splats along their major tangent axis. Child
  centers/covariances preserve the parent's first two mixture moments; opacity
  approximately preserves integrated optical density, not exact rendered pixels.
  Children inherit their parent's material and rigid motion group.
- Remove only opacity below 0.008. Strong isolated geometry is not blindly erased.
- Independently denoise V3's original Mesh with bounded Taubin iterations and
  bilateral vertex colors, protecting topology, face orientation and group motion.
  Recompute binding offsets to preserve the original canonical attachment centers.
- In the viewer, refined V3 uses the existing depth/alpha-guided 3×3 interior-only
  surface filter. This last step is display filtering, distinct from geometry.

The surface-alignment motivation is related to
[SuGaR](https://arxiv.org/abs/2311.12775), but this implementation is a conservative
local geometry prior, **not a reproduction of SuGaR training**. It does not implement
Mip-Splatting's learned sampling-aware filters.

## Reproduce

```bash
python scripts/fan/refine_fan_surface.py \
  --manifest <original-fan-manifest.json> --output <new-gaussian-directory> \
  --neighbors 256 --strength 1 --color-strength 1 --maximum-shift 0.003

python scripts/mesh/denoise_viewer_mesh.py \
  --input <original-right-mesh.npz> --output <new-mesh-directory> \
  --iterations 32 --color-iterations 6

python scripts/fan/package_fan_surface.py \
  --manifest <new-gaussian-directory>/manifest.json \
  --mesh <new-mesh-directory>/mesh.npz --output <new-package-directory>

python scripts/fan/evaluate_fan_surface.py \
  --manifest <new-package-directory>/manifest.json --output <new-evaluation-directory> \
  --frames 0,25,50,75,99 --resolution 1024 --display-smoothing

python -m unittest tests.fan.test_fan_surface tests.fan.test_fan_mesh tests.mesh.test_mesh_surface
```

The optional `surface_refinement` entry in the Fan manifest enables the derived
asset only on the right/V3 side. The original sources and their fingerprints
remain unchanged. Loading validates source/binding/mesh hashes, parent identities,
motion groups and displacement bounds. Removing this entry restores original GS
behavior; use the previous manifest to restore the original Mesh as well.

## 2026-10-24 validation

906,401 → 1,203,676 Gaussians; 300,000 split parents; 2,725 negligible-opacity
points removed. 689,685 original centers have a supported local surface fit.
No new observations are introduced by splitting. Mesh rebinding maximum center
error is 4.22e-10 in source coordinates.

All image comparisons use the previous V3 as baseline, identical motion, cameras,
resolution and lighting. They measure **display proxies, not geometric accuracy**:

| Output | Samples | Interior Laplacian reduction | Minimum silhouette IoU |
| --- | --- | --- | --- |
| GS, geometry + display filter | 6 views × 5 times | 19.11% | 99.20% |
| White clay, geometry + display filter | 6 views × 5 times | 8.34% | 99.20% |
| Pseudocolor, geometry + display filter | 6 views × 5 times | 8.77% | 99.20% |
| Pure Mesh | 6 views × 3 times | 15.64% | 99.84% |

Average strong-edge gradient ratios for GS/clay/pseudocolor are 95.78%, 96.27%,
96.35%; the worst samples are 89.48%, 90.31%, 89.55%. Smoothing therefore has a
measurable sharpness tradeoff and is not evidence that every detail improves.
The earlier geometry-only six-view/frame-zero probe reduced GS Laplacian by 7.00%.
Raw reports, model candidates and previous assets are kept outside Git in the
server's `scgs-results/fan-surface-20261024` directory.

Deployment regression checks passed: all five modes across six views (eight real
frames per view); sampled frame 50 V1 outputs in all five modes and the Lego
pseudocolor pair are pixel-identical to the previous deployment. The final Fan
pseudocolor viewer also cached all 100 real frames for all six views at 1024 px.
After disconnecting the browser network, each view completed a 105-presented-frame
loop at 30 FPS with zero dropped frames and zero buffer underruns. Render requests
remained 600 before/after the offline loops. This is one tested browser/device,
not a guarantee for every client. Checks are under `outputs/fan-surface-20261024`.
