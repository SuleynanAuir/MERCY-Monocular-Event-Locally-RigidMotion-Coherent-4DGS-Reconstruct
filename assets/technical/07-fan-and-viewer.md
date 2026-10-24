# Fan 视频拟合与表面细化

Fan 是独立示例，输入来源和版本含义与 Lego 不同。其基础运动来自 `Fan.mkv` 的转角拟合，而非 Blender Object 动画；后续的绑定修复和表面细化也不构成事件监督运动重建。已有原始事件不完整记录与显示用事件视频必须分别理解。

## 1 视频拟合刚体运动

[prepare_fan_viewer.py](../../scripts/fan/prepare_fan_viewer.py) 读取初始高斯、给定相机与视频，利用世界 Z 轴、近似平面叶片及前层分组等显式假设，估计右侧叶片逐帧转角。左侧风扇及后方支架保持静止。

运行输出 `fan-motion.npz` 和 `motion-report.json`，记录高斯、运动、中心、分组、拟合误差和输入来源。检查规范帧、旋转中心、部件归属、角度曲线与视频投影。该模型自由度较低，不能代表任意三维非刚体运动恢复。

```bash
python scripts/fan/prepare_fan_viewer.py \
  --data /path/to/Fan_data --output outputs/fan-video-viewer
```

## 2 修复旋转组误绑定

[repair_fan_motion_binding.py](../../scripts/fan/repair_fan_motion_binding.py) 使用视频多时刻支持度检查“静止位置更吻合、旋转位置不吻合”的高斯，将误归入旋转组的支架点恢复为静止归属。

```bash
python scripts/fan/repair_fan_motion_binding.py \
  --sequence outputs/fan-video-viewer/fan-motion.npz \
  --motion-report outputs/fan-video-viewer/motion-report.json \
  --data /path/to/Fan_data --output outputs/fan-binding-fixed
```

输出保存修复序列及重分配行号。此阶段改变归属，保持其余已有转角和高斯外观；应检查支架突出物和叶片完整性。偶数帧选择、奇数帧检查只是一种分组支持度验证，因为转角拟合已使用完整视频，不能称为整个重建的独立测试。

## 3 规范表面细化

后续表面流程见 [实现说明](../../scripts/fan/SURFACE_REFINEMENT.md) 与 [fan_surface.py](../../utils/fan_surface.py)。处理规范高斯时保持原有刚体运动来源：

1. 在同一运动组内建立邻域，用法向一致性、颜色相似性和稳健高度筛选拟合局部平面，保护边界、细线和病态邻域。
2. 对得到支持的高斯施加有界法向位移、朝向对齐和局部 DC 颜色正则，避免跨组或跨层拉平。
3. 将受支持的大 splat 沿主切向分裂，子高斯继承父材料和运动组；位置与协方差按混合矩保持，透明度仅近似保持积分光学密度，不保证渲染像素精确不变。
4. 仅清理低于阈值的微弱透明度项，不把强不透明孤立结构自动当作噪声删除。
5. 分别处理原有 V3 Mesh，保护拓扑与三角面朝向，并重算绑定偏移以保持原规范附着中心。
6. 显示端可另施加深度／alpha 引导的内部表面滤波。这一步是显示处理，与前面的几何修改分开评估。

新流程不读取 RGB 帧进行训练，也没有引入额外观测。它是受限局部几何先验，不是 SuGaR 训练的复现；不能借用其他方法名称推定实现了其损失或采样模型。

## 4 打包与回退

```bash
python scripts/fan/refine_fan_surface.py \
  --manifest /path/to/original-fan-manifest.json \
  --output outputs/fan-surface \
  --neighbors 256 --strength 1 --color-strength 1 --maximum-shift 0.003

python scripts/mesh/denoise_viewer_mesh.py \
  --input /path/to/original-right-mesh.npz \
  --output outputs/fan-surface-mesh --iterations 32 --color-iterations 6

python scripts/fan/package_fan_surface.py \
  --manifest outputs/fan-surface/manifest.json \
  --mesh outputs/fan-surface-mesh/mesh.npz --output outputs/fan-surface-package
```

`surface_refinement` 配置只启用右侧 V3 派生资产；加载器检查源、绑定、Mesh 哈希及运动组与位移范围。恢复原 GS 可移除此项；恢复原 Mesh 应使用之前的完整 manifest，不能只更换一个文件而保留不匹配绑定。

## 5 评估与结果范围

```bash
python scripts/fan/evaluate_fan_surface.py \
  --manifest outputs/fan-surface-package/manifest.json \
  --output outputs/fan-surface-evaluation \
  --frames 0,25,50,75,99 --resolution 1024 --display-smoothing
```

仓库 2026-10-24 实现说明记录：906,401 个高斯变为 1,203,676 个，300,000 个父高斯被分裂，2,725 个低透明度项被清理。增加点数不等于获得新几何观测。

| 显示代理，比较前后 V3 | 采样 | 内部 Laplacian 降低 | 最低轮廓 IoU |
| --- | --- | ---: | ---: |
| GS，几何＋显示滤波 | 6 视角 × 5 时刻 | 19.11% | 99.20% |
| 白模，几何＋显示滤波 | 6 视角 × 5 时刻 | 8.34% | 99.20% |
| 伪彩色，几何＋显示滤波 | 6 视角 × 5 时刻 | 8.77% | 99.20% |
| Mesh | 6 视角 × 3 时刻 | 15.64% | 99.84% |

这些是图像代理而非几何真值准确度。GS／白模／伪彩色平均强边缘梯度比分别为 95.78%、96.27%、96.35%，说明去颗粒伴有锐度损失。记录来自既有报告，本次文档更新没有重新部署或复跑。

## 6 截图、参考视频与布局

README 的两张 Fan 截图均为第 25 帧附近的 0.833 s 显示，采用不同观察视角，按一行两列展示。原图中的 V1/V3 是 Fan 的历史界面标签；不能将其中 V3 标为 Lego MERCY 的事件训练结果。

界面中 RGB／事件参考可与动画帧同步，但参考事件视频不替代原始事件数组。伪彩色基于初始位置着色，不是材质、深度真值或事件误差。白底合成、自动留白与内部去噪是显示配置，不能混入训练评估。

完整查看器说明见 [操作参考](09-viewer-reference.md)，统一评估要求见 [指标与消融](06-evaluation.md)。
