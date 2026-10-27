# Original UI across multiple local GPU forwards

The experiment panel requires the updated GPU viewer and this proxy. It forwards
`POST /api/experiments/plan`, `POST /api/experiments`, job polling and streaming ZIP
downloads to the selected object's service. See
[experiment exports](../../assets/technical/13-experiment-export.md) for reference
calibration and independent evaluation. Rendering packets remain unchanged.

The comparison button opens `/comparison` in a new tab on this same unified UI.
Its page, CSS/JS, `/api/comparisons` jobs and nested report/download assets use
`asset_upstream`, independently of the selected scene. Multipart uploads stream
through the proxy with their original boundary, including files larger than the
8 MiB JSON request limit. Report security headers are preserved for same-origin
iframe previews.

Workflow: first save all experiment data in the rendering UI. Then select the
resulting archive under **existing exports** in the comparison page (no download
and re-upload needed), enter its server path, or upload a local ZIP / extracted
experiment folder. ZIP uploads do not need manual extraction.

`serve_multi_object_ui.py` serves the existing viewer HTML, JavaScript and CSS
without modification. It merges the **original object dropdown** and forwards
each selected object's HTTP/WebSocket requests to its configured GPU service.
An optional `revision_label` appends a provenance label to the right-side display
labels in the status response. Render packets pass through unchanged. It introduces no portal, replay fallback,
replacement model or image enhancement.

```bash
pip install -r requirements/viewer.txt
python scripts/viewer/serve_multi_object_ui.py --catalog /path/to/catalog.json --port 18896
```

Keep the GPU viewers and SSH local forwards running. Example catalog:

```json
{
  "default": "lamp",
  "asset_upstream": "http://127.0.0.1:18893",
  "objects": [
    {"id": "lego", "name": "Lego · 吊车（重建版本）", "revision_label": "重建版本", "pending": true, "note": "按原 V3 流程重建中。"},
    {"id": "fan", "name": "Fan · 风扇", "live": ["http://127.0.0.1:18894"]},
    {"id": "lamp", "name": "Lamp · 台灯", "live": ["http://127.0.0.1:18894"]},
    {"id": "metronome", "name": "Metronome · 节拍器", "live": ["http://127.0.0.1:18893"]},
    {"id": "microphone", "name": "Microphone · 麦克风", "live": ["http://127.0.0.1:18893"]}
  ]
}
```

Open `http://127.0.0.1:18896/?scene=lamp`. An object without `live` stays visible
in the menu and reports its missing assets explicitly; it cannot render until
its genuine assets are restored. `pending: true` keeps the original loading
screen active while reconstruction is running; it does not mark the scene ready.
Do not advertise an object as ready before its renderer has loaded successfully.
Adding a catalog entry does not itself run reconstruction or model repair.
The catalog is reread for each request; the UI's existing object switching clears
old camera/frame caches by loading a new document.

Only loopback upstreams are accepted. Host/Origin checks protect the local entry
point. No arbitrary local files or remote destinations are served.

Validation: `python -m unittest tests.viewer.test_multi_object_ui`.

For Lego restoration, retain the original damaged
`Lego_data/fan_contrastive_g0/contrastive_g0.ply` as V1. Use the original
`ComparisonRenderer` and freshly audited surface-repair/completion artifacts
for V3. Preserve source, pose, correction and event-audit checksums. An old
audit cannot be attached to newly generated assets. Keep `重建版本` in the
object name when the historical checkpoint was regenerated, and add `live`
only after checking actual same-frame, same-camera render packets.

对照试验结果按 `experiment/output/物品名_YYYYMMDD_HHMMSS_ffffff/` 保存（北京时间），PNG 和 ZIP 均带同一前缀。渲染页按钮旁的物品名自动传入评测页。可启动 `python -m scripts.viewer.serve_comparison_ui --port 8896`，再在 catalog 设置本机转发地址 `comparison_upstream`；无需重载 GPU 模型。旧报告 ID 和链接继续有效。
