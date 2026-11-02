# lamp — 推荐保留 V3

推荐连续表面/覆盖修复V3，优先整体几何覆盖；早期外观版PSNR更高但IoU较低，不将两版数值拼接。

样本数：25。来源：`outputs/lamp-recovery-20261030/server/reports/lamp_recovered_20261030_224817_518682/data/analysis.json`。
原始文件 SHA256：`26cd58320a1a888528594f091898944368a97a808937e6e2f261a2ccb765e41b`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/26cd58320a1a888528594f091898944368a97a808937e6e2f261a2ccb765e41b.json) · [最佳版本登记](../best/lamp.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 17.098771985746787 | 18.335278906140722 | 21.48543740153816 |
| psnr | dB ↑ | 23.4424007584315 | 25.221931552934144 | 30.63749934961496 |
| ssim | ↑ | 0.9382496405827655 | 0.9604149098001912 | 0.9716795152804569 |
| foreground_ssim | ↑ | 0.6699398221277134 | 0.7676956122900261 | 0.789668105664862 |
| lpips | ↓ | 0.06456219762563706 | 0.0495651450753212 | 0.029175426587462427 |
| iou | ↑ | 0.8162726366471514 | 0.8755861485136001 | 0.966323135146142 |
| missing | % ↓ | 3.947340793350743 | 5.081353405635217 | 1.7307794718454923 |
| part_missing | % ↓ | 2.4949159464222634 | 4.369124800288545 | 2.0648082845175075 |
| event_l1 | ↓ | 1.1254897917558229 | 0.8672401645244059 | 0.5257844665609134 |
| event_cosine | ↑ | 0.5234825630986549 | 0.6888734289133099 | 0.8511300690594001 |
| polarity | % ↑ | 70.02524054180248 | 77.83228890353601 | 84.01138174540318 |
| silent | ↓ | 0.013609674861633813 | 0.006913700377530183 | 0.0024825600862946008 |
| latency_ms | ms ↓ | 19.958783239126205 | 41.77106411755085 | 31.67855717241764 |
| point_count | points | 426697 | 853403 | 426697 |
| metric_unavailable_reasons | 依源协议 | {} | {} | {} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.44671130180358887 | 0.38660693168640137 | 0.442173033952713 |
| position_jerk_rms | 依源协议 | 6.623364873841169 | 7.075947226940933 | 6.545028441150532 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 426697 | 853403 | 426697 |
| mean_ms | 依源协议 | 19.958783239126205 | 41.77106411755085 | 31.67855717241764 |
| p50_ms | 依源协议 | 9.924553334712982 | 20.928801968693733 | 12.194868177175522 |
| p95_ms | 依源协议 | 75.57682283222675 | 88.22412192821503 | 74.23658072948456 |

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
  "mean_ms": "96.74004732072353",
  "p50_ms": "96.26874886453152",
  "p95_ms": "177.88302041590214",
  "comparison_samples_per_second": "10.336980678587917",
  "load_seconds": "30.22952800616622",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "431484416",
  "versions": {
    "V1": {
      "mean_ms": "19.958783239126205",
      "p50_ms": "9.924553334712982",
      "p95_ms": "75.57682283222675",
      "frames_per_second": "50.103254693384805"
    },
    "V2": {
      "mean_ms": "41.77106411755085",
      "p50_ms": "20.928801968693733",
      "p95_ms": "88.22412192821503",
      "frames_per_second": "23.94001735713102"
    },
    "V3": {
      "mean_ms": "31.67855717241764",
      "p50_ms": "12.194868177175522",
      "p95_ms": "74.23658072948456",
      "frames_per_second": "31.567094251082082"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "aaef92688be989899eacdd6ec870fc6b6612bae2e0cb12ab3c382caac1f61566",
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
| [experiment/output/lamp_20261027_205903_366488/data/analysis.json](../sources/63eed47006d813fb863b5dd303d6f007b967215e31ef726e937e0827c1eae660.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261027_205903_366488/data/analysis.json](../sources/63eed47006d813fb863b5dd303d6f007b967215e31ef726e937e0827c1eae660.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261027_205903_366488/data/analysis.json](../sources/63eed47006d813fb863b5dd303d6f007b967215e31ef726e937e0827c1eae660.json) | V3 | 28.389669139335005 | 0.9692924487308288 | 0.04377747967839241 | 0.8797755974978755 |
| [experiment/output/lamp_20261029_011355_185757/data/analysis.json](../sources/b5943cc5cb5f6b08647d9720cd0f384355638ac1fdc35446b83154d2d4f7305c.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261029_011355_185757/data/analysis.json](../sources/b5943cc5cb5f6b08647d9720cd0f384355638ac1fdc35446b83154d2d4f7305c.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261029_011355_185757/data/analysis.json](../sources/b5943cc5cb5f6b08647d9720cd0f384355638ac1fdc35446b83154d2d4f7305c.json) | V3 | 35.32712668034959 | 0.9856210207694392 | 0.012447962313890457 | 0.8801760023689311 |
| [experiment/output/lamp_20261029_012553_501321/data/analysis.json](../sources/0d103199f36f40e71b1de934ee557bfd33093ad406dc264949a5d814970c5f79.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261029_012553_501321/data/analysis.json](../sources/0d103199f36f40e71b1de934ee557bfd33093ad406dc264949a5d814970c5f79.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261029_012553_501321/data/analysis.json](../sources/0d103199f36f40e71b1de934ee557bfd33093ad406dc264949a5d814970c5f79.json) | V3 | 35.272012736064724 | 0.9866923994367198 | 0.011448489911854267 | 0.8801760023689311 |
| [experiment/output/lamp_20261029_025138_116921/data/analysis.json](../sources/43ecd155490c3b8e9d245ee3d2bcfde5a76473627cbfd305fe3e396060ed4415.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261029_025138_116921/data/analysis.json](../sources/43ecd155490c3b8e9d245ee3d2bcfde5a76473627cbfd305fe3e396060ed4415.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261029_025138_116921/data/analysis.json](../sources/43ecd155490c3b8e9d245ee3d2bcfde5a76473627cbfd305fe3e396060ed4415.json) | V3 | 26.67144039309616 | 0.9572746608088433 | 0.036417896151542666 | 0.8242775346979971 |
| [experiment/output/lamp_20261029_025748_848652/data/analysis.json](../sources/5b2b9463f54d799c1f238f833797aa2b5ed019bc80362ab378518da9428ce27f.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261029_025748_848652/data/analysis.json](../sources/5b2b9463f54d799c1f238f833797aa2b5ed019bc80362ab378518da9428ce27f.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261029_025748_848652/data/analysis.json](../sources/5b2b9463f54d799c1f238f833797aa2b5ed019bc80362ab378518da9428ce27f.json) | V3 | 24.7722329536439 | 0.9433668612370544 | 0.05510564863681793 | 0.8242775346979971 |
| [experiment/output/lamp_20261029_035616_805220/data/analysis.json](../sources/873dd20ad8f1bd93f18d9f9a4425fab64fdbd29e7aa119f95c3356057f7ad295.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [experiment/output/lamp_20261029_035616_805220/data/analysis.json](../sources/873dd20ad8f1bd93f18d9f9a4425fab64fdbd29e7aa119f95c3356057f7ad295.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [experiment/output/lamp_20261029_035616_805220/data/analysis.json](../sources/873dd20ad8f1bd93f18d9f9a4425fab64fdbd29e7aa119f95c3356057f7ad295.json) | V3 | 30.63749934961496 | 0.9716795152804569 | 0.029175426587462427 | 0.966323135146142 |
| [outputs/lamp-recovery-20261030/server/reports/lamp_recovered_20261030_224817_518682/data/analysis.json](../sources/26cd58320a1a888528594f091898944368a97a808937e6e2f261a2ccb765e41b.json) | V1 | 23.4424007584315 | 0.9382496405827655 | 0.06456219762563706 | 0.8162726366471514 |
| [outputs/lamp-recovery-20261030/server/reports/lamp_recovered_20261030_224817_518682/data/analysis.json](../sources/26cd58320a1a888528594f091898944368a97a808937e6e2f261a2ccb765e41b.json) | V2 | 25.221931552934144 | 0.9604149098001912 | 0.0495651450753212 | 0.8755861485136001 |
| [outputs/lamp-recovery-20261030/server/reports/lamp_recovered_20261030_224817_518682/data/analysis.json](../sources/26cd58320a1a888528594f091898944368a97a808937e6e2f261a2ccb765e41b.json) | V3 | 30.63749934961496 | 0.9716795152804569 | 0.029175426587462427 | 0.966323135146142 |
