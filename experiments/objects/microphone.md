# microphone — 推荐保留 V3

本地完整三版本实验中推荐V3；模型外观推荐不等于速度最优。

样本数：25。来源：`experiment/output/microphone_20261027_210649_602833/data/analysis.json`。
原始文件 SHA256：`a752bcb401f3de812ce47d698683c7440a813cfa75b3987cdb5f7b59278c7e59`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/a752bcb401f3de812ce47d698683c7440a813cfa75b3987cdb5f7b59278c7e59.json) · [最佳版本登记](../best/microphone.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 16.013819810291537 | 21.12089491200756 | 25.499077374221496 |
| psnr | dB ↑ | 25.606164247410348 | 30.556464864811005 | 34.37805633737692 |
| ssim | ↑ | 0.942793802827864 | 0.9687635736681124 | 0.9836985003348273 |
| foreground_ssim | ↑ | 0.4997247075946675 | 0.6716538865441842 | 0.8387917297604893 |
| lpips | ↓ | 0.06626810774207115 | 0.030467469841241837 | 0.018658003583550454 |
| iou | ↑ | 0.8334430023956391 | 0.9202617481575588 | 0.9234937668365624 |
| missing | % ↓ | 3.2359005974860287 | 3.7263646862734516 | 3.6046463973548466 |
| part_missing | % ↓ | 5.819575175757154 | 6.138758159594203 | 5.913567327855031 |
| event_l1 | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_cosine | ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| polarity | % ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| silent | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| latency_ms | ms ↓ | 31.478596806526184 | 47.765218675136566 | 53.179174304008484 |
| point_count | points | 227244 | 454420 | 454420 |
| metric_unavailable_reasons | 依源协议 | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} | {"event_l1": "缺少原始事件；彩色预览视频不能还原事件计数"} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.4180070459842682 | 0.42222920060157776 | 0.42222920060157776 |
| position_jerk_rms | 依源协议 | 71.21214898917002 | 89.62676095496123 | 89.62676095496123 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 227244 | 454420 | 454420 |
| mean_ms | 依源协议 | 31.478596806526184 | 47.765218675136566 | 53.179174304008484 |
| p50_ms | 依源协议 | 18.27400177717209 | 30.19396960735321 | 63.16686421632767 |
| p95_ms | 依源协议 | 61.919230222702026 | 82.84853249788284 | 83.01037102937698 |

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
  "mean_ms": "136.51414877176285",
  "p50_ms": "126.83525681495667",
  "p95_ms": "171.29331082105637",
  "comparison_samples_per_second": "7.325248034706599",
  "load_seconds": "25.059684917330742",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "514444800",
  "versions": {
    "V1": {
      "mean_ms": "31.478596806526184",
      "p50_ms": "18.27400177717209",
      "p95_ms": "61.919230222702026",
      "frames_per_second": "31.767616776128936"
    },
    "V2": {
      "mean_ms": "47.765218675136566",
      "p50_ms": "30.19396960735321",
      "p95_ms": "82.84853249788284",
      "frames_per_second": "20.935735829061205"
    },
    "V3": {
      "mean_ms": "53.179174304008484",
      "p50_ms": "63.16686421632767",
      "p95_ms": "83.01037102937698",
      "frames_per_second": "18.804353641959857"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "d0175f3d95e71b04743e18f02869bc2354d93afee892df450170385ef33a2b29",
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
| [experiment/output/microphone_20261027_210649_602833/data/analysis.json](../sources/a752bcb401f3de812ce47d698683c7440a813cfa75b3987cdb5f7b59278c7e59.json) | V1 | 25.606164247410348 | 0.942793802827864 | 0.06626810774207115 | 0.8334430023956391 |
| [experiment/output/microphone_20261027_210649_602833/data/analysis.json](../sources/a752bcb401f3de812ce47d698683c7440a813cfa75b3987cdb5f7b59278c7e59.json) | V2 | 30.556464864811005 | 0.9687635736681124 | 0.030467469841241837 | 0.9202617481575588 |
| [experiment/output/microphone_20261027_210649_602833/data/analysis.json](../sources/a752bcb401f3de812ce47d698683c7440a813cfa75b3987cdb5f7b59278c7e59.json) | V3 | 34.37805633737692 | 0.9836985003348273 | 0.018658003583550454 | 0.9234937668365624 |
