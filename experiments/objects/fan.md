# fan — 推荐保留 V3

推荐周期外观V3；相比早期V3图像与事件表现更优，但早期V3的IoU略高，保留其原始记录。

样本数：50。来源：`experiment/output/fan_20261029_094351_952280/data/analysis.json`。
原始文件 SHA256：`00e5822d6bc0ac5a42087284cfb3063db3c73abde188ae80d8ffe43b2a566bf4`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/00e5822d6bc0ac5a42087284cfb3063db3c73abde188ae80d8ffe43b2a566bf4.json) · [最佳版本登记](../best/fan.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 18.568993604886945 | 18.617399242213306 | 24.022734831949286 |
| psnr | dB ↑ | 23.91488578016638 | 24.237982226224837 | 29.763931415327978 |
| ssim | ↑ | 0.9106152206133519 | 0.9134780634767681 | 0.9730666831017706 |
| foreground_ssim | ↑ | 0.6955787188693155 | 0.6964865175400026 | 0.9232645448866552 |
| lpips | ↓ | 0.047266949824988844 | 0.043743849229067565 | 0.016968038380146027 |
| iou | ↑ | 0.9570977773902 | 0.9656726924991039 | 0.9656726924991039 |
| missing | % ↓ | 2.165998803308047 | 2.1462883093985745 | 2.1462883093985745 |
| part_missing | % ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_l1 | ↓ | 0.7568113107529639 | 0.6921038321193609 | 0.41429993073794336 |
| event_cosine | ↑ | 0.7236422238785576 | 0.7619708563948339 | 0.8755456083407751 |
| polarity | % ↑ | 73.61186512407647 | 73.70258228504844 | 87.35221722935094 |
| silent | ↓ | 0.018634668875947648 | 0.01500321236443131 | 0.007604171352926579 |
| latency_ms | ms ↓ | 28.136076271533966 | 25.244730055332184 | 29.83078521490097 |
| point_count | points | 906401 | 906401 | 906401 |
| metric_unavailable_reasons | 依源协议 | {} | {} | {} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.5986508727073669 | 0.5986508727073669 | 0.5986508727073669 |
| position_jerk_rms | 依源协议 | 5.799340609610142 | 5.751036953101139 | 5.751036953101139 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 906401 | 906401 | 906401 |
| mean_ms | 依源协议 | 28.136076271533966 | 25.244730055332184 | 29.83078521490097 |
| p50_ms | 依源协议 | 13.277057558298111 | 13.305503875017166 | 13.777010142803192 |
| p95_ms | 依源协议 | 72.95494116842745 | 75.01568980515002 | 71.22931145131587 |

## 源报告：性能范围

```json
{
  "operation": "benchmark",
  "export_operation": "complete",
  "timing_file": "performance/timings.json",
  "samples": "250",
  "warmup": "20",
  "repeats": "5",
  "scope": "serial render of all panels, GPU synchronization and CPU pixel transfer; no JPEG cache",
  "includes_state_capture": false,
  "mean_ms": "87.2113007903099",
  "p50_ms": "90.47006070613861",
  "p95_ms": "112.40714006125924",
  "comparison_samples_per_second": "11.466403905663457",
  "load_seconds": "204.8290711119771",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "944312320",
  "versions": {
    "V1": {
      "mean_ms": "28.136076271533966",
      "p50_ms": "13.277057558298111",
      "p95_ms": "72.95494116842745",
      "frames_per_second": "35.541558472804084"
    },
    "V2": {
      "mean_ms": "25.244730055332184",
      "p50_ms": "13.305503875017166",
      "p95_ms": "75.01568980515002",
      "frames_per_second": "39.61222789105563"
    },
    "V3": {
      "mean_ms": "29.83078521490097",
      "p50_ms": "13.777010142803192",
      "p95_ms": "71.22931145131587",
      "frames_per_second": "33.52241628223998"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "cf37ab0fff5ca1d25bc1155b189cf2d9a4551d184d0c5ed6da4e6c6a20ef94cf",
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
  "sample_count": "50",
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
    "48",
    "50",
    "52",
    "54",
    "56",
    "58",
    "60",
    "62",
    "64",
    "66",
    "68",
    "70",
    "72",
    "74",
    "76",
    "78",
    "80",
    "82",
    "84",
    "86",
    "88",
    "90",
    "92",
    "94",
    "96",
    "98"
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
| [experiment/output/fan_20261027_211809_635553/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V1 | 23.91488578016638 | 0.9106152206133519 | 0.047266949824988844 | 0.9570977773902 |
| [experiment/output/fan_20261027_211809_635553/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V2 | 24.237982226224837 | 0.9134780634767681 | 0.043743849229067565 | 0.9656726924991039 |
| [experiment/output/fan_20261027_211809_635553/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V3 | 26.465077501545416 | 0.9426350094775506 | 0.0262449249625206 | 0.969465448695934 |
| [experiment/output/fan_20261029_092758_703479/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V1 | 23.91488578016638 | 0.9106152206133519 | 0.047266949824988844 | 0.9570977773902 |
| [experiment/output/fan_20261029_092758_703479/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V2 | 24.237982226224837 | 0.9134780634767681 | 0.043743849229067565 | 0.9656726924991039 |
| [experiment/output/fan_20261029_092758_703479/data/analysis.json](../sources/42a62bb07dabb3301e575d86f4f321bba635ac6dceaf51e19096164582ce4c69.json) | V3 | 26.465077501545416 | 0.9426350094775506 | 0.0262449249625206 | 0.969465448695934 |
| [experiment/output/fan_20261029_093610_872329/data/analysis.json](../sources/3de0a643d94407107f459c1bd8a282e4d36f7cbd02384b8e45a3b8ef889aabce.json) | V1 | 23.91488578016638 | 0.9106152206133519 | 0.047266949824988844 | 0.9570977773902 |
| [experiment/output/fan_20261029_093610_872329/data/analysis.json](../sources/3de0a643d94407107f459c1bd8a282e4d36f7cbd02384b8e45a3b8ef889aabce.json) | V2 | 24.237982226224837 | 0.9134780634767681 | 0.043743849229067565 | 0.9656726924991039 |
| [experiment/output/fan_20261029_093610_872329/data/analysis.json](../sources/3de0a643d94407107f459c1bd8a282e4d36f7cbd02384b8e45a3b8ef889aabce.json) | V3 | 29.763931415327978 | 0.9730666831017704 | 0.016968038380146027 | 0.9656726924991039 |
| [experiment/output/fan_20261029_094351_952280/data/analysis.json](../sources/00e5822d6bc0ac5a42087284cfb3063db3c73abde188ae80d8ffe43b2a566bf4.json) | V1 | 23.91488578016638 | 0.9106152206133519 | 0.047266949824988844 | 0.9570977773902 |
| [experiment/output/fan_20261029_094351_952280/data/analysis.json](../sources/00e5822d6bc0ac5a42087284cfb3063db3c73abde188ae80d8ffe43b2a566bf4.json) | V2 | 24.237982226224837 | 0.9134780634767681 | 0.043743849229067565 | 0.9656726924991039 |
| [experiment/output/fan_20261029_094351_952280/data/analysis.json](../sources/00e5822d6bc0ac5a42087284cfb3063db3c73abde188ae80d8ffe43b2a566bf4.json) | V3 | 29.763931415327978 | 0.9730666831017706 | 0.016968038380146027 | 0.9656726924991039 |
