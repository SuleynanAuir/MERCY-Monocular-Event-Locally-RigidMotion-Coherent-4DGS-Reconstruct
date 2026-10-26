# 复现步骤与产物接口

本页给出 Lego 融合路线的依赖顺序和已有入口。命令是可填写路径的运行模板，不声称仓库自带全部数据或已完成本轮训练。输出使用新目录；不同版本的基础高斯、绑定与运动缓存不可混用。

## 1 准备环境与数据

训练和渲染需要 CUDA GPU、匹配的 PyTorch/CUDA、编译工具及本地 rasterizer／simple-knn；涉及节点操作的模块还依赖 PyTorch3D。Blender 导出需 Blender Python 环境，参考视频处理需 FFmpeg／ffprobe，Mesh 流程参照 `requirements/mesh.txt`。

AutoDL 安装入口是 [setup_autodl.sh](../../scripts/setup/setup_autodl.sh)，手动安装按 [README](../../readme.md) 执行。使用环境脚本前确认实际解释器与数据盘路径。环境配置和模型重建是两个步骤，成功安装不等于数据有效。

核对 G0 首帧身份、视频 FPS、事件列顺序与时间单位、相机分辨率、Blender 帧采样。记录输入 SHA-256、脚本版本和实际命令。原始事件数组与用于界面的事件预览视频应分别命名。

## 2 导出 Blender 部件

```bash
blender --background --disable-autoexec /path/to/Lego.blend \
  --python scripts/motion/export_blender_parts.py -- \
  --output outputs/lego-init/blender-surface-parts.npz \
  --frames 400 --start 1 --step 10
```

帧数、起点和步长必须与实际数据一致；上例是历史 Lego 配置的模板，不是通用默认。`--disable-autoexec` 关闭场景内嵌自动执行；显式导出脚本读取求值后的对象和动画。导出后检查部件列表、采样时间、动态矩阵和弹性表面的顶点身份。

## 3 诊断相机与 G0

```bash
python scripts/diagnostics/inspect_lego_data.py \
  --data /path/to/Lego_data --output outputs/lego-inspection
```

该脚本读取 `Lego.jpg` 作为首帧参考，同时比较 `Lego.ply` 与 `fan_contrastive_g0/contrastive_g0.ply`；这两个诊断输入不包含在 README 的最小融合数据示意中。`fixed_camera.json` 还需具有脚本读取的 R、T、宽高、fy、背景与观察距离字段。它输出首帧渲染和 `inspection.json`，不是重新训练 G0。若自己的目录不符合脚本预期，应明确适配读取配置，不能换用来源不同的模型来凑齐输入。

## 4 建立绑定与基础缓存

```bash
python scripts/motion/bind_lego_parts.py \
  --data /path/to/Lego_data \
  --parts outputs/lego-init/blender-surface-parts.npz \
  --inspection outputs/lego-inspection/inspection.json \
  --output outputs/lego-init/initial-binding.npz

python scripts/motion/cache_lego_initial_motion.py \
  --data /path/to/Lego_data --initialization outputs/lego-init \
  --camera outputs/lego-inspection/inspection.json \
  --output outputs/lego-baseline
```

缓存脚本的 `--camera` 读取含 `camera` 字段的报告，不应直接替换为任意外参 JSON。若使用独立修正 G0，缓存可显式给出 `--source-ply`，但绑定必须先匹配同一 G0 的行序与几何，不能只在缓存阶段更换模型。

基础输出含 `poses.npy`、`dynamic_indices.npy`、渲染缓存及元数据。逐帧检查首帧恒等、绑定归属和刚体内距离。该步骤形成 Bind-GS 基线。

## 5 准备事件监督与关节报告

```bash
python scripts/events/prepare_lego_events.py \
  --data /path/to/Lego_data --output outputs/lego-event-supervision

python scripts/diagnostics/diagnose_lego_joints.py \
  --parts outputs/lego-init/blender-surface-parts.npz \
  --output outputs/lego-joint-report.json
```

监督输出含 `supervision.npz` 和 `calibration.json`。关节诊断输出路径是报告文件，不是目录。检查标定可靠性、留出区间、关节层级与闭环；诊断完成并不自动保证所有连接都满足训练假设。

## 6 优化硬关节校正

```bash
python scripts/events/refine_lego_temporal.py \
  --baseline outputs/lego-baseline \
  --supervision outputs/lego-event-supervision/supervision.npz \
  --parts outputs/lego-init/blender-surface-parts.npz \
  --binding outputs/lego-init/initial-binding.npz \
  --joint-report outputs/lego-joint-report.json \
  --foreground-weight 12 --lr 0.01 --iterations 800 \
  --output outputs/lego-event-hard-joint
```

这是含结构先验的训练分支，不是纯事件或原始 SC-GS 基线。保存训练目标、校正配置、完整审计与父来源。若仅评估 Align-GS 中间模块，应明确是否开放关节约束，避免与 MERCY 混入不同配置。

## 7 几何与外观阶段

根据诊断选择连贯绑定、刚性表面修复、多视角先验补全和外观细化工具，入口见 [MERCY 流程](03-mercy.md)。这些工具依赖各自的模型、掩码、监督或先验归档，不能只用一次运动校正输出替代全部输入。

每次新增或筛除高斯，重新记录行号映射、动态组和父来源；外观优化前冻结声明不变的参数。完成后重新做事件、覆盖和序列化检查。历史细节增强、Mesh 和受保护表面命令另见 [补充处理入口](10-processing-recipes.md)。

## 8 局部 MLP

```bash
python scripts/deformation/train_v3_deformation.py \
  --base-field /path/to/verified-base-field.pth \
  --steps 300 --output outputs/lego-local-mlp
```

`base-field` 是带来源、绑定及几何信息的兼容基础场，不是任意 `deform.pth`，也不是单独的 PLY。当前入口复用局部训练器，序列化保留历史 `v4_local_joint_mlp_v1` 格式。候选须经过真实渲染复核；局部评估入口见 `scripts/deformation/evaluate_v4_focused_local.py`。

## 9 导出、配置与重载

```bash
python scripts/events/export_lego_event_preview.py \
  --baseline outputs/lego-baseline \
  --candidate outputs/lego-event-hard-joint \
  --output outputs/lego-event-preview
```

严格验收不通过时默认导出会拒绝；仅为研究预览，可显式增加 `--allow-unapproved-preview`，同时保留失败状态。这个开关不使候选通过验收。

普通序列、带几何／外观层的比较产物与局部辅助配置不是同一文件。高级候选应使用对应打包入口和 comparison manifest；旧单序列入口可能拒绝会丢失细化层的产物。配置 `deformation.json` 时使用真实 checkpoint、audit 和哈希，不手工填入伪造来源。

```bash
python viewer.py --scenes_path /path/to/scenes.json \
  --host 127.0.0.1 --port 8892
```

用同帧同相机比较重载前后的结果，核对辅助变形没有重复叠加、两侧运动没有串用。界面能播放不等于模型通过研究验收。

## 产物接口

| 产物 | 核心内容 | 必须一致的来源 |
| --- | --- | --- |
| 部件归档 | 表面、标签、Object 矩阵、变形顶点、时间 | Blender 场景与采样设置 |
| 绑定归档 | 标签、最近点、动态行号、配准 | G0 行序与部件归档 |
| 基础姿态 | XYZ／四元数的逐帧缓存 | G0、绑定、相机与时间 |
| 事件监督 | 灰度、计数、标定、区间划分 | 原始视频和事件 |
| 校正 JSON | 模型格式、参数、关节信息 | 基础姿态及监督 |
| 细化／补全层 | 几何、外观、归属和父哈希 | 完整父模型与校正 |
| MLP 权重及配置 | 网络、冻结条件、增益、审计引用 | 同一父 V3 与事件口径 |
| 查看器 manifest | 序列、左右模型、Mesh、参考、辅助 | 每侧独立资产及对应时间 |

## 数组布局与坐标检查

| 数据 | 当前路径中的典型布局 | 容易混淆的点 |
| --- | --- | --- |
| Object 矩阵 | 帧 × 部件 × 4 × 4 | 保存绝对世界矩阵，应用前计算相对首帧变换 |
| 基础 `poses.npy` | 帧 × 动态高斯 × 7 | XYZ 后接 wxyz；动态行号映射至完整 G0 |
| 导出序列 `poses` | 动态高斯 × 帧 × 7 | 与基础缓存的前两轴顺序不同，不能直接替换 |
| 事件监督灰度 | 帧 × 高 × 宽 | 由视频解码生成，使用元数据解释亮度范围 |
| 区间事件计数 | 区间 × 2 × 高 × 宽 | 正事件在前、负事件在后；区间数通常比帧数少一 |
| 部件标签与映射 | 按高斯或表面行号保存 | 新增／筛除高斯后必须同步更新，不能只复用旧长度 |

这些布局对应本文引用的代码路径；读取其他格式时先核对其元数据。校验 shape 只能发现部分问题，还要检查行号身份、时间、四元数顺序、有限值及父来源。

## 故障定位

| 现象 | 优先检查 |
| --- | --- |
| 首帧已经错位 | G0 身份、相机、配准及帧起点 |
| 运动后断臂或拖影 | 错绑、未绑定、表面覆盖、父姿态匹配 |
| 事件损失很大 | 时间单位、窗口偏移、极性、亮度下限与标定 |
| 角轨迹变平但动作变小 | 幅度、相位、锚点及覆盖约束 |
| 重载后不同 | 文件哈希、动态行号、四元数布局和重复补偿 |
| 网页卡顿 | 服务端渲染、传输、浏览器解码与缓存；勿直接归因于轨迹 |

返回 [技术索引](README.md)。
