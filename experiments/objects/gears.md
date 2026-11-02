# gears — 推荐保留 V3

推荐紧致感知V3：大位移候选PSNR/IoU更高，但侧背面粗糙；当前紧位移版LPIPS更好。保留全部候选，不声称单项分数全局最高。

样本数：25。来源：`outputs/six-object-preview/gears-compact-quality-20261102/analysis.json`。
原始文件 SHA256：`1d78963aa8b5d2c0619284f11c6a55b5f21216f716b8a0e54fe2875ec9fc4755`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/1d78963aa8b5d2c0619284f11c6a55b5f21216f716b8a0e54fe2875ec9fc4755.json) · [最佳版本登记](../best/gears.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 17.731469423205077 | 16.0341292514275 | 22.272410431067932 |
| psnr | dB ↑ | 22.922128867533903 | 21.475424853896293 | 26.904059104718353 |
| ssim | ↑ | 0.8689474854530056 | 0.8596931182043004 | 0.9367989195756833 |
| foreground_ssim | ↑ | 0.46835707623218975 | 0.41562865302251417 | 0.8057704072403398 |
| lpips | ↓ | 0.08478252187371255 | 0.10004515558481217 | 0.030549167469143868 |
| iou | ↑ | 0.7889955270264405 | 0.7881986174400233 | 0.7978299336181484 |
| missing | % ↓ | 11.593685593112298 | 7.766114295868383 | 5.669175339247538 |
| part_missing | % ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_l1 | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_cosine | ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| polarity | % ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| silent | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| latency_ms | ms ↓ | 40.83147529512644 | 13.997030965983868 | 18.16778402775526 |
| point_count | points | 828050 | 467436 | 301739 |
| metric_unavailable_reasons | 依源协议 | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.7338931560516357 | 0.7593529224395752 | 0.7940030097961426 |
| position_jerk_rms | 依源协议 | 3919.5959421756893 | 3362.8329615872917 | 3670.009261795888 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 828050 | 467436 | 301739 |
| mean_ms | 依源协议 | 40.83147529512644 | 13.997030965983868 | 18.16778402775526 |
| p50_ms | 依源协议 | 16.68041292577982 | 7.656832225620747 | 8.294145576655865 |
| p95_ms | 依源协议 | 78.079195600003 | 57.74730062112211 | 66.11103285104036 |

## 源报告：性能范围

```json
{
  "operation": "benchmark",
  "export_operation": "complete",
  "timing_file": "performance/timings.json",
  "samples": "50",
  "warmup": "2",
  "repeats": "2",
  "scope": "serial render of all panels, GPU synchronization and CPU pixel transfer; no JPEG cache",
  "includes_state_capture": false,
  "mean_ms": "77.20142673701048",
  "p50_ms": "88.7324120849371",
  "p95_ms": "96.14145206287503",
  "comparison_samples_per_second": "12.95312848824073",
  "load_seconds": "22.713606797158718",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "414174720",
  "versions": {
    "V1": {
      "mean_ms": "40.83147529512644",
      "p50_ms": "16.68041292577982",
      "p95_ms": "78.079195600003",
      "frames_per_second": "24.49091032768434"
    },
    "V2": {
      "mean_ms": "13.997030965983868",
      "p50_ms": "7.656832225620747",
      "p95_ms": "57.74730062112211",
      "frames_per_second": "71.44372277451119"
    },
    "V3": {
      "mean_ms": "18.16778402775526",
      "p50_ms": "8.294145576655865",
      "p95_ms": "66.11103285104036",
      "frames_per_second": "55.04248611015419"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "ea244e8df25121acdb820da59b75901902abf44e6bba0b071cad6aaf0b09cbd4",
  "source_version_ids": [
    "0",
    "1",
    "2"
  ],
  "version_order": [
    "V1",
    "V2",
    "V3"
  ],
  "sample_count": "25",
  "resolutions": [
    [
      "640",
      "640"
    ]
  ],
  "source_frames": [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24"
  ],
  "event_primary_rule": "response_validation only; calibration-overlap and unspecified windows retained in raw metrics but excluded",
  "bootstrap": "5000 paired stationary resamples, blocks 5/10/20; descriptive within sequence",
  "scope": "Comparison only; no ablations",
  "version_presentation": {
    "revision": "swap-v1-v2-20261027",
    "source_to_display": {
      "V1": "V2",
      "V2": "V1",
      "V3": "V3"
    },
    "disclosure": "版本重新编号：V1=原V2，V2=原V1，V3不变；模型、测量值与原始来源不变，不代表新增训练或已验证逐层消融。",
    "changes": "Labels and display/export ordering only; raw assets and measured values unchanged"
  }
}
```

## 历史候选（不同协议不直接排名）

| 来源 | 版本 | PSNR | SSIM | LPIPS | IoU |
|---|---|---|---|---|---|
| [outputs/six-object-preview/gears-compact-quality-20261102/analysis.json](../sources/1d78963aa8b5d2c0619284f11c6a55b5f21216f716b8a0e54fe2875ec9fc4755.json) | V1 | 22.922128867533903 | 0.8689474854530056 | 0.08478252187371255 | 0.7889955270264405 |
| [outputs/six-object-preview/gears-compact-quality-20261102/analysis.json](../sources/1d78963aa8b5d2c0619284f11c6a55b5f21216f716b8a0e54fe2875ec9fc4755.json) | V2 | 21.475424853896293 | 0.8596931182043004 | 0.10004515558481217 | 0.7881986174400233 |
| [outputs/six-object-preview/gears-compact-quality-20261102/analysis.json](../sources/1d78963aa8b5d2c0619284f11c6a55b5f21216f716b8a0e54fe2875ec9fc4755.json) | V3 | 26.904059104718353 | 0.9367989195756833 | 0.030549167469143868 | 0.7978299336181484 |
| [outputs/six-object-preview/gears-geometry-quality-20261102/analysis.json](../sources/1058cdd88d64194a606a6039e3a9f214f0317da8e02c3f479e581cbf22f24c9a.json) | V1 | 22.922128867533903 | 0.8689474854530056 | 0.08478252187371255 | 0.7889955270264405 |
| [outputs/six-object-preview/gears-geometry-quality-20261102/analysis.json](../sources/1058cdd88d64194a606a6039e3a9f214f0317da8e02c3f479e581cbf22f24c9a.json) | V2 | 21.475424853896293 | 0.8596931182043004 | 0.10004515558481217 | 0.7881986174400233 |
| [outputs/six-object-preview/gears-geometry-quality-20261102/analysis.json](../sources/1058cdd88d64194a606a6039e3a9f214f0317da8e02c3f479e581cbf22f24c9a.json) | V3 | 29.885218482331684 | 0.952416252863652 | 0.03170002847909927 | 0.8425251894526022 |
| [outputs/six-object-preview/gears-quality-20261102/analysis.json](../sources/c4403d97f93d05188fbfd57175bb40819c25c951293878f39243ba6341a485a3.json) | V1 | 22.922128867533903 | 0.8689474854530056 | 0.08478252187371255 | 0.7889955270264405 |
| [outputs/six-object-preview/gears-quality-20261102/analysis.json](../sources/c4403d97f93d05188fbfd57175bb40819c25c951293878f39243ba6341a485a3.json) | V2 | 21.475424853896293 | 0.8596931182043004 | 0.10004515558481217 | 0.7881986174400233 |
| [outputs/six-object-preview/gears-quality-20261102/analysis.json](../sources/c4403d97f93d05188fbfd57175bb40819c25c951293878f39243ba6341a485a3.json) | V3 | 26.226376652800738 | 0.9303946945244351 | 0.03520059034228325 | 0.7813925146845477 |
