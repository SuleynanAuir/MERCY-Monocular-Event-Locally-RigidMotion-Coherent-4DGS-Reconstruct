# arm — 推荐保留 V3

推荐RGB结构优化V3；5时刻留出结果另存，正式13帧含训练/选点重叠，不是独立测试。

样本数：13。来源：`outputs/six-object-preview/arm-rgb-optimized-result.json`。
原始文件 SHA256：`1c78d1d7a7ef2ffa50efcf960aafd58ec4096d7b99d17486c3c8a994f549a496`；JSON pointer：`/summary/2`。

[原始文件快照](../sources/1c78d1d7a7ef2ffa50efcf960aafd58ec4096d7b99d17486c3c8a994f549a496.json) · [最佳版本登记](../best/arm.json)

以下数值直接使用源JSON十进制数值字符串，未四舍五入、重算或补零。

| 指标（源字段） | 单位/方向 | V1 | V2 | V3 |
|---|---|---|---|---|
| foreground_psnr | dB ↑ | 17.787373389264648 | 16.651627966125606 | 24.79716915354198 |
| psnr | dB ↑ | 26.358016679529207 | 25.389878362812777 | 33.14267325096382 |
| ssim | ↑ | 0.9373096013523357 | 0.9366108625125824 | 0.9785752880255936 |
| foreground_ssim | ↑ | 0.5152684463119312 | 0.5209347133139121 | 0.8660097830341229 |
| lpips | ↓ | 0.048299386810797915 | 0.06996332108974457 | 0.015282241054452382 |
| iou | ↑ | 0.8662287694746728 | 0.8555949741193761 | 0.9219734350362097 |
| missing | % ↓ | 7.09069067428794 | 7.069771155278108 | 1.8327004199329981 |
| part_missing | % ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_l1 | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| event_cosine | ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| polarity | % ↑ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| silent | ↓ | 缺失（未填零） | 缺失（未填零） | 缺失（未填零） |
| latency_ms | ms ↓ | 33.71795073438149 | 27.934847614512993 | 36.58462086549172 |
| point_count | points | 576117 | 288103 | 538488 |
| metric_unavailable_reasons | 依源协议 | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} | {"event_l1": "缺少可靠事件标定或完整时间窗：Raw event file is not available on this host"} |

## 源警告

- V3 使用同一源视角的 RGB 监督优化；本报告 13 个时刻中 10 个参与训练、3 个用于留出验证及检查点选择。不是独立测试集或新视角泛化证明。
- 前景 PSNR / IoU 使用原始 RGB 的固定背景分割参考，属于光度前景代理掩码，不是人工或几何真值；阈值与定义随实验包保存。

## 历史候选（不同协议不直接排名）

| 来源 | 版本 | PSNR | SSIM | LPIPS | IoU |
|---|---|---|---|---|---|
| [outputs/six-object-preview/arm-evaluation-result.json](../sources/070531809b20c87b1ba107cbd787aac0097cd9d8b54432ecd7423639eeb92a19.json) | V1 | 26.358016679529207 | 0.9373096013523357 | 0.048299386810797915 | 0.8662287694746728 |
| [outputs/six-object-preview/arm-evaluation-result.json](../sources/070531809b20c87b1ba107cbd787aac0097cd9d8b54432ecd7423639eeb92a19.json) | V2 | 25.389878362812777 | 0.9366108625125824 | 0.06996332108974457 | 0.8555949741193761 |
| [outputs/six-object-preview/arm-evaluation-result.json](../sources/070531809b20c87b1ba107cbd787aac0097cd9d8b54432ecd7423639eeb92a19.json) | V3 | 26.321204713742585 | 0.9611812274764711 | 0.03587696930536857 | 0.9206615048048424 |
| [outputs/six-object-preview/arm-rgb-optimized-result.json](../sources/1c78d1d7a7ef2ffa50efcf960aafd58ec4096d7b99d17486c3c8a994f549a496.json) | V1 | 26.358016679529207 | 0.9373096013523357 | 0.048299386810797915 | 0.8662287694746728 |
| [outputs/six-object-preview/arm-rgb-optimized-result.json](../sources/1c78d1d7a7ef2ffa50efcf960aafd58ec4096d7b99d17486c3c8a994f549a496.json) | V2 | 25.389878362812777 | 0.9366108625125824 | 0.06996332108974457 | 0.8555949741193761 |
| [outputs/six-object-preview/arm-rgb-optimized-result.json](../sources/1c78d1d7a7ef2ffa50efcf960aafd58ec4096d7b99d17486c3c8a994f549a496.json) | V3 | 33.14267325096382 | 0.9785752880255936 | 0.015282241054452382 | 0.9219734350362097 |
| [outputs/six-object-preview/arm-rgb-validation-result.json](../sources/a34a38cced1c638f362b08618c66c8426a8cde1cb39512fe51a403a6f21b990c.json) | V1 | 26.119882724065484 | 0.9360558821156569 | 0.049563823640346526 | 0.858340552777656 |
| [outputs/six-object-preview/arm-rgb-validation-result.json](../sources/a34a38cced1c638f362b08618c66c8426a8cde1cb39512fe51a403a6f21b990c.json) | V2 | 25.27279637979999 | 0.9358152523167929 | 0.07062409073114395 | 0.8497837636560549 |
| [outputs/six-object-preview/arm-rgb-validation-result.json](../sources/a34a38cced1c638f362b08618c66c8426a8cde1cb39512fe51a403a6f21b990c.json) | V3 | 32.39232168643901 | 0.9765203625795177 | 0.01704017408192158 | 0.914156863606447 |
