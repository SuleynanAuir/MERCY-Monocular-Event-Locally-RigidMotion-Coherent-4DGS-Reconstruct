# rotation — 推荐保留 V3

推荐连续时序外观V3；8个留出时刻七项均值也优于V1/V2，属于同源视角选点验证。

样本数：40。来源：`outputs/six-object-preview/rotation-quality-20261102/analysis.json`。
原始文件 SHA256：`052b3f75d3f3574329ff0ac917c8a389f2b1a8226c7b8b52f1b275aa6cd252b9`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/052b3f75d3f3574329ff0ac917c8a389f2b1a8226c7b8b52f1b275aa6cd252b9.json) · [最佳版本登记](../best/rotation.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 18.290442056706887 | 17.13749525258616 | 31.21816687766296 |
| psnr | dB ↑ | 24.807764343856945 | 24.149478215781777 | 38.70829694548269 |
| ssim | ↑ | 0.89955503314858 | 0.8945281973158217 | 0.9896410610496869 |
| foreground_ssim | ↑ | 0.49841282823614197 | 0.4642859913052394 | 0.9504345447833284 |
| lpips | ↓ | 0.0770137382671237 | 0.08006673585623503 | 0.0072767836099956185 |
| iou | ↑ | 0.8802938360677928 | 0.8710691286731551 | 0.9624986159707006 |
| missing | % ↓ | 3.9439523106846908 | 4.181979690157121 | 0.08106015002258349 |
| part_missing | % ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_l1 | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_cosine | ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| polarity | % ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| silent | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| latency_ms | ms ↓ | 41.94731707684696 | 27.748005208559334 | 32.097523333504796 |
| point_count | points | 1567112 | 1155218 | 1016971 |
| metric_unavailable_reasons | 依源协议 | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} |

## 源报告：motion

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 256 | 256 | 256 |
| unassigned_points | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| motion_amplitude | 依源协议 | 0.38592860102653503 | 0.42187315225601196 | 0.40928295254707336 |
| position_jerk_rms | 依源协议 | 72.79254018914418 | 65.70426501842812 | 68.64063208343347 |
| rotation_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| angular_jerk_rms | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| joint_gap_max | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| rigidity_error | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| swing_max_rad | 依源协议 | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |

## 源报告：efficiency

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| points | 依源协议 | 1567112 | 1155218 | 1016971 |
| mean_ms | 依源协议 | 41.94731707684696 | 27.748005208559334 | 32.097523333504796 |
| p50_ms | 依源协议 | 32.69384242594242 | 13.842753134667873 | 23.11662118881941 |
| p95_ms | 依源协议 | 78.01041826605797 | 58.91167195513843 | 68.53899508714676 |

## 源报告：性能范围

```json
{
  "operation": "benchmark",
  "export_operation": "complete",
  "timing_file": "performance/timings.json",
  "samples": "80",
  "warmup": "2",
  "repeats": "2",
  "scope": "serial render of all panels, GPU synchronization and CPU pixel transfer; no JPEG cache",
  "includes_state_capture": false,
  "mean_ms": "104.13743339013308",
  "p50_ms": "98.79960864782333",
  "p95_ms": "151.4699226245284",
  "comparison_samples_per_second": "9.602694895059216",
  "load_seconds": "72.00926424004138",
  "memory_scope": "PyTorch process-wide allocations; includes all loaded scenes, not per-version memory",
  "memory_peak": "830214656",
  "versions": {
    "V1": {
      "mean_ms": "41.94731707684696",
      "p50_ms": "32.69384242594242",
      "p95_ms": "78.01041826605797",
      "frames_per_second": "23.83942692134547"
    },
    "V2": {
      "mean_ms": "27.748005208559334",
      "p50_ms": "13.842753134667873",
      "p95_ms": "58.91167195513843",
      "frames_per_second": "36.03862665023334"
    },
    "V3": {
      "mean_ms": "32.097523333504796",
      "p50_ms": "23.11662118881941",
      "p95_ms": "68.53899508714676",
      "frames_per_second": "31.15505173435473"
    }
  }
}
```

## 源报告：实验协议

```json
{
  "manifest_sha256": "2c429294ebedf58393af3e5c38649d737810ef334c3ceac728d40bb9e8903753",
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
  "sample_count": "40",
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
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39"
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
| [outputs/six-object-preview/rotation-quality-20261102/analysis.json](../sources/052b3f75d3f3574329ff0ac917c8a389f2b1a8226c7b8b52f1b275aa6cd252b9.json) | V1 | 24.807764343856945 | 0.89955503314858 | 0.0770137382671237 | 0.8802938360677928 |
| [outputs/six-object-preview/rotation-quality-20261102/analysis.json](../sources/052b3f75d3f3574329ff0ac917c8a389f2b1a8226c7b8b52f1b275aa6cd252b9.json) | V2 | 24.149478215781777 | 0.8945281973158217 | 0.08006673585623503 | 0.8710691286731551 |
| [outputs/six-object-preview/rotation-quality-20261102/analysis.json](../sources/052b3f75d3f3574329ff0ac917c8a389f2b1a8226c7b8b52f1b275aa6cd252b9.json) | V3 | 38.70829694548269 | 0.9896410610496869 | 0.0072767836099956185 | 0.9624986159707006 |
