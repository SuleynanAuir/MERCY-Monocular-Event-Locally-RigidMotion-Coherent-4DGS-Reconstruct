# Event-reference repairs — 2026-11-04

These files preserve empirical response calibrations from original RGB and raw events. Calibration and reported response-validation windows are disjoint. This is not independent sensor calibration or evidence of unseen-view generalization.

- Arm, Robot, Gears and Rotation: calibration records archived here; their earlier server reports are not included in this snapshot.
- Fan, Metronome and Microphone: calibration records, exact report summaries and verification counts in `result.json`; repair and verification scripts from the completed server runs.
- Fan also includes source-geometry versus RGB foreground overlap diagnostics. Metronome and Microphone preserve existing source-part references.

The scripts capture the original server paths and export fingerprints. Run from the server repository root with the matching source exports and reference caches; output directories must be new. `repair.py prepare` creates references and `repair.py report` launches comparison evaluation. `verify.py` checks original image metrics and independently recomputes part means and event summary means. Raw events, videos, meshes, model weights and full reports remain outside Git.

The source renders are unchanged. The repairs supply missing reference measurements; they do not retrain the models. Existing `experiments/best` snapshots retain their original provenance and are not silently replaced.
