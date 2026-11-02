# metronome — 推荐保留 V3

本地完整三版本实验中推荐V3；保留原始速度和运动指标，不要求这些指标同步最优。

样本数：25。来源：`experiment/output/metronome_20261027_210532_030915/data/analysis.json`。
原始文件 SHA256：`5df55a784066977aa26d1f65e78ecba0c9d6839a062d017f35da8b4c4f4ad79d`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/5df55a784066977aa26d1f65e78ecba0c9d6839a062d017f35da8b4c4f4ad79d.json) · [最佳版本登记](../best/metronome.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 22.87003197794122 | 28.03238441281433 | 34.705598197476924 |
| psnr | dB ↑ | 32.53989025240102 | 37.07680065504374 | 43.245488620437115 |
| ssim | ↑ | 0.969982634069803 | 0.9792445704971532 | 0.9950098880132784 |
| foreground_ssim | ↑ | 0.7305862673048762 | 0.8218611706419554 | 0.9720274995733557 |
| lpips | ↓ | 0.040099219977855684 | 0.032009847611188885 | 0.013377347327768803 |
| iou | ↑ | 0.9576692757856445 | 0.9588325670077963 | 0.9619114684053561 |
| missing | % ↓ | 1.0264941680426254 | 1.7095260219400585 | 1.4871858633599546 |
| part_missing | % ↓ | 38.68971085493154 | 37.03940073935731 | 31.764644197027 |
| event_l1 | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_cosine | ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| polarity | % ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| silent | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| latency_ms | ms ↓ | 18.357515454292297 | 64.90013313293457 | 68.70008605718613 |
| point_count | points | 645262 | 1143789 | 1143789 |
| metric_unavailable_reasons | 依源协议 | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.4546181559562683 | 0.3470359146595001 | 0.3470359146595001 |
| position_jerk_rms | 依源协议 | 6.141159884506379 | 25.349525833369 | 25.349525833369 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 645262 | 1143789 | 1143789 |
| mean_ms | 依源协议 | 18.357515454292297 | 64.90013313293457 | 68.70008605718613 |
| p50_ms | 依源协议 | 13.8038769364357 | 56.490905582904816 | 72.6177915930748 |
| p95_ms | 依源协议 | 43.99837851524352 | 88.45673352479935 | 92.16839075088501 |

## 源报告：性能范围

```json
{
  "operation": "benchmark",
  "export_operation": "complete",
  "timing_file": "performance/timings.json",
  "samples": "125",
  "warmup": "20",
  "repeats": "5",
  "scope": "serial render of all panels, GPU synchronization and CPU pixel transfer; no JPEG cache",
  "includes_state_capture": false,
  "mean_ms": "156.1957774758339",
  "p50_ms": "153.73555570840836",
  "p95_ms": "193.29882860183716",
  "comparison_samples_per_second": "6.402221725582286",
  "load_seconds": "31.763014279305935",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "681384960",
  "versions": {
    "V1": {
      "mean_ms": "18.357515454292297",
      "p50_ms": "13.8038769364357",
      "p95_ms": "43.99837851524352",
      "frames_per_second": "54.47360251390572"
    },
    "V2": {
      "mean_ms": "64.90013313293457",
      "p50_ms": "56.490905582904816",
      "p95_ms": "88.45673352479935",
      "frames_per_second": "15.40828888519698"
    },
    "V3": {
      "mean_ms": "68.70008605718613",
      "p50_ms": "72.6177915930748",
      "p95_ms": "92.16839075088501",
      "frames_per_second": "14.556022523284723"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "298b7a785cfadb16373229a6db2bd802bba515e057987178f2fce21502dbae3c",
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
    "2",
    "4",
    "6",
    "8",
    "10",
    "12",
    "14",
    "16",
    "18",
    "20",
    "22",
    "24",
    "26",
    "28",
    "30",
    "32",
    "34",
    "36",
    "38",
    "40",
    "42",
    "44",
    "46",
    "48"
  ],
  "event_primary_rule": "response_validation only; calibration-overlap and unspecified windows retained in raw metrics but excluded",
  "bootstrap": "5000 paired stationary resamples, blocks 5/10/20; descriptive within sequence",
  "scope": "Comparison only; no ablations",
  "version_presentation": {
    "revision": "swap-v1-v2-20261027",
    "source_to_display": {
      "V1": "V1",
      "V2": "V2",
      "V3": "V3"
    },
    "disclosure": "版本按阶段清单排列；保留原始来源与哈希，指标以同相机同时间实测为准。",
    "changes": "Labels and display/export ordering only; raw assets and measured values unchanged"
  }
}
```

## 历史候选（不同协议不直接排名）

| 来源 | 版本 | PSNR | SSIM | LPIPS | IoU |
|---|---|---|---|---|---|
| [experiment/output/metronome_20261027_210532_030915/data/analysis.json](../sources/5df55a784066977aa26d1f65e78ecba0c9d6839a062d017f35da8b4c4f4ad79d.json) | V1 | 32.53989025240102 | 0.969982634069803 | 0.040099219977855684 | 0.9576692757856445 |
| [experiment/output/metronome_20261027_210532_030915/data/analysis.json](../sources/5df55a784066977aa26d1f65e78ecba0c9d6839a062d017f35da8b4c4f4ad79d.json) | V2 | 37.07680065504374 | 0.9792445704971532 | 0.032009847611188885 | 0.9588325670077963 |
| [experiment/output/metronome_20261027_210532_030915/data/analysis.json](../sources/5df55a784066977aa26d1f65e78ecba0c9d6839a062d017f35da8b4c4f4ad79d.json) | V3 | 43.245488620437115 | 0.9950098880132784 | 0.013377347327768803 | 0.9619114684053561 |
