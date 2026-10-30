# Lamp V3 加载回退修复 · 2026-10-30

本次问题是服务加载到了旧候选。8898（浏览）和 8908（实验）都使用 `scgs-lamp-continuous-20261029/scenes.json`，该清单仍指向同目录的旧 `pair.json`。其 V3 为 1,706,788 个高斯、四倍切向细分和冻结表面的时序外观拟合；选优只比较自身初值，没有要求超过 V2。用户给出的 12 项数值与这一候选的历史报告一致。

服务器已有经过部件内连续几何、时序外观和覆盖优化的 `scgs-lamp-compact-20261029/overlap-pair.json`。本次核验其清单、模型和外观文件哈希与历史通过报告一致后，将上述服务清单指向该候选，只重启两个 lamp 服务。这是恢复已有改进模型，未新增训练、改写 V1/V2 或更改评测参考。原始配置备份在服务器 `/root/autodl-tmp/scgs-lamp-recovery-20261030/scenes-before.json`；启动时继续读取原场景清单也会加载正确候选。

改进清单 SHA256：`e4772acbb4e53996d5dbd97008a5081b5ef52870781a1c31130428e049b2ef89`。
外观与几何检查点 SHA256：`bd47cfd5560afc8a44353820f7a54b5eeaeb7c0bf27f7e1ea2b6a8f7cc0cbfa3`。
V3 高斯数为 426,697，保留原部件动画。

通过当前实验服务重新导出 640×640、固定参考相机、黑背景、每 2 帧采样的 25 帧实验，再重新计算全部指标：

| 指标 | V2 | 修复后 V3 |
|---|---:|---:|
| Foreground PSNR ↑ | 18.33528 | 21.48544 |
| Full-frame PSNR ↑ | 25.22193 | 30.63750 |
| Full-frame SSIM ↑ | 0.960415 | 0.971680 |
| Foreground SSIM ↑ | 0.767696 | 0.789668 |
| LPIPS ↓ | 0.049565 | 0.029175 |
| Silhouette IoU ↑ | 0.875586 | 0.966323 |
| Foreground missing % ↓ | 5.08135 | 1.73078 |
| Visible-part missing % ↓ | 4.36912 | 2.06481 |
| Event relative L1 ↓ | 0.867240 | 0.525784 |
| Event cosine ↑ | 0.688873 | 0.851130 |
| Polarity agreement % ↑ | 77.83229 | 84.01138 |
| No-event log-change ↓ | 0.006914 | 0.002483 |

同一开发序列的 12 项均值均优于 V2；这不是独立序列或新视角泛化验证。模型使用 RGB 和公共掩码监督，不能称为事件单独监督。已有候选经过多轮开发集比较，这组完整序列指标也不能作为无偏测试集结果。

新导出：`outputs/observation-exports/lamp_20261030T144756Z_63d907f9.zip`（服务器项目内）。新报告：本地 `outputs/lamp-recovery-20261030/server/reports/lamp_recovered_20261030_224817_518682/report.html`。251 个渲染图像与参考文件和已验证候选历史导出逐字节一致；新报告独立重算了指标。两个采样计划 JSON 并非逐字节相同，因此不声称完整导出文件完全一致。

本地 20896 网关的 live/observations 通道分别检查了 Gaussian、clay、pseudocolor，共 6 张三版本渲染图，位于 `outputs/lamp-recovery-20261030/`。人工查看了斜视角 Gaussian 预览。浏览器页面交互未做自动化操作。

新增只读发布预检脚本，核验报告完整性、候选依赖哈希与 12 项质量指标；任一指标缺失、非有限值或低于 V2 均拒绝。它是显式运行的预检，不是已安装的服务启动钩子：

```bash
python scripts/lamp/verify_release.py \
  --pair /root/autodl-tmp/scgs-lamp-compact-20261029/overlap-pair.json \
  --report /root/autodl-tmp/scgs-lamp-recovery-20261030/reports/lamp_recovered_20261030_224817_518682
python -m unittest tests.viewer.test_lamp_release -v
```

5 项测试通过，覆盖事件退步、无改善、缺失/NaN/Inf 和模型/清单指纹不匹配。部署记录和新评测摘要见 [15-lamp-recovery.json](15-lamp-recovery.json)。
