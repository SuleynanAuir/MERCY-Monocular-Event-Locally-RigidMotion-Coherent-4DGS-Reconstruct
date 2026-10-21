# MonoEvent-RT3D-DMR

面向 Lego 机械结构的动态三维高斯重建：从早期运动不稳定、局部断臂的问题出发，结合视频帧、事件流与结构约束，改善运动连续性、部件完整性和多视角细节。

本项目按照三阶段对比展开：

**原始 SC-GS：运动不稳定 → 改进版 1：增加视频帧监督，仍有断臂 → 改进版 2：视频帧 + 事件流，并结合结构约束，追求细节清晰、全方位结构完整、关节运动连续。**

这里的“乱动”“断臂”描述本项目早期运行中的定性观察，不是对原始 SC-GS 在所有场景下的结论。“增加视频帧监督”指本项目的改进配置，并非声称原始 SC-GS 不使用图像。三组统一配置的定量消融尚待补齐；下面明确区分已有结果、方法目标和额外先验。

## 三阶段方法对比

| 阶段 | 本项目的输入与约束变化 | 重点观察 | 当前结论与边界 |
| --- | --- | --- | --- |
| A · 原始 SC-GS | 原始方法配置；未加入本项目的事件校正与机械关节完整性流程 | 是否出现非预期运动、局部漂移和时序不稳定，即“乱动” | 本项目早期使用反馈；需保留具体配置与同时间点结果后再做定量比较 |
| B · 改进版 1：视频帧监督 | 在项目基线上增加视频帧监督，不使用事件流作为监督 | 外观与帧内位置之外，机械臂是否仍有断裂、部件连接是否连续 | 已反馈局部断臂；帧拟合好不能直接证明三维结构完整 |
| C · 改进版 2：视频帧 + 事件流 | 视频提供外观参照，事件约束时间变化；结合刚体分组、硬关节连接和表面修复 | 帧间运动、机械臂连续性、细节以及正面 / 侧面 / 背面效果 | 已有融合实验支持局部断裂与覆盖改善；现有实现还使用 Blender 几何和运动初始化先验，不能把全部提升归因于事件流 |

下文目录名中的 `v1`、`v3` 等是各子实验内部编号，不等同于上表的“改进版 1 / 2”。现有融合演示也不是“只改一个事件开关”的三组消融。

### A · 原始 SC-GS：从“能动”到“动得合理”的问题

本项目早期结果出现非预期运动。对 Lego 这样的机械结构，观察重点不仅是高斯是否随时间变化，还包括同一部件是否保持形状、相连部件是否脱开，以及运动方向和时序是否与观测一致。

因此，原始结果作为对照的重点是**运动稳定性与结构一致性**，而不是仅展示一个能够播放的动画。

### B · 改进版 1：增加视频帧监督，但仍出现局部断臂

视频帧为外观和可见位置提供监督，使重建有明确的图像参照；但单帧投影接近，并不自动保证机械臂在三维空间中连续连接。此前观察到的断臂、局部缺口与部件错位，需要单独检查。

这一阶段要回答的是：**仅靠视频帧，能否同时约束外观、帧间运动和部件连接？** 本项目的局部断臂反馈说明，仅展示帧内外观还不够。

### C · 改进版 2：视频帧 + 事件流，联合改善运动、结构与细节

融合流程针对不同问题使用不同约束：

- **运动连续性**：视频帧提供外观参照，原始事件流约束帧间对数亮度变化；通过真实高斯渲染检查运动，而不是直接播放输入视频。
- **机械臂完整性**：同组刚体高斯共享旋转和平移，硬关节前向运动学保持连接点一致；连贯的表面重建与覆盖保护用于减少局部碎裂和缺口。事件监督本身不等于自动补齐几何。
- **细节清晰度**：在冻结已建立的几何和运动后，进行局部外观优化，检查机械臂、齿轮、驾驶室等区域的纹理、边缘与亮度误差。
- **全方位结构检查**：从正面、侧面、背面及留出视角检查覆盖。现有背面补全来自显式 Blender 网格先验，不是单目视频和事件直接观测到的真实背面。

目标是得到**运动有依据、部件不断开、局部细节清晰、自由视角下结构更完整**的动态高斯。已有抽检中，主梁与连杆的早期大面积碎裂明显减少；小缺口、软化和未通过的事件区间仍保留记录，不宣称所有时间、所有视角都已经“零断臂”。

### “符合物理原理”具体指什么？

当前已实现和检查的是**刚体与关节的运动学一致性**：

1. 同一刚体组内的高斯随同一个刚体变换运动，避免把刚性部件任意拉长。
2. 关节两侧使用同一个连接点；树状关节由前向运动学约束，闭环连接按源数据精度检查，并限制相关校正自由度。
3. 使用平滑时序参数和事件误差检查运动连续性，但平滑本身不构成运动正确的证据。

历史 `lego-multiview-v2` 的结构检查覆盖 400 帧、每帧 14 个关节：最大关节间隙为 `1.42e-6` 场景单位，低于 `1e-5` 的检查门槛；每帧抽检的刚体点对距离最大误差为 `8.94e-8` 场景单位。这支持“所检刚体与关节保持几何一致”，不等于所有渲染像素都没有缺口。

当前没有完成力、力矩、质量、惯性、碰撞或能量守恒的动力学验证；弹性带也没有新增真实弹性求解。因此本文不将运动学约束表述为完整物理仿真。

## 已有证据与三组对照的展示方式

以下是融合流程内部的分阶段证据，**不是 A / B / C 三组统一条件的定量消融**：

| 检查项 | 已有记录 | 可以支持的结论 |
| --- | --- | --- |
| 机械臂可见部件覆盖 | 表面修复前后，40 个采样帧中满足条件的部件帧有 1,489 个；其中缺失超过 10% 的部件帧由 130 个降到 0 个，仍有 56 个缺失像素 | 所检亮前景内部的大面积缺失减少；不覆盖所有边界、暗部或隐藏表面 |
| 背面覆盖 | 运动刚体背面的留出视角检查（alpha 阈值 0.5），像素加权缺失率由 12.9274% 降到 0.3481% | 给定几何先验下覆盖改善，不代表全部静态部件或未知背面真实细节被恢复 |
| 关节连接 | 400 帧结构检查，最大关节间隙 `1.42e-6` 场景单位 | 已检查关节的运动学连接保持在容差内 |
| 事件一致性 | `lego-global-detail-v1` 为 382 / 399 个区间通过，仍有 17 个失败区间 | 候选尚未通过全序列严格事件验收，不能宣称完整无误 |

来源为本地审计记录 `outputs/lego-comparison/fracture-surface-inclusive/evidence-summary.json`、`outputs/lego-multiview-v2-independent/heldout-coverage.json`、`outputs/lego-multiview-v2-reports/structure-check.json` 及 `outputs/lego-global-detail-v1/reports/`；大型产物和完整报告不随 Git 仓库发布。不同记录对应不同实验阶段，不能拼接成同一个模型已经通过全部检查。

正式三组对照应使用同一场景、时间戳、相机、分辨率和可比训练预算，并明确初始高斯及额外先验是否相同，排列为 **A 原始 SC-GS / B 视频帧 / C 视频帧 + 事件流**，分别展示：

- **整体运动**：完整运动周期，观察方向、相位、漂移和抖动。
- **机械臂局部**：同一时刻放大关节和细杆，观察断臂、错位、缺口及拉伸。
- **多视角**：固定同一时间，比较正面、侧面和背面；背面必须注明是否使用几何先验。
- **定量检查**：视频误差、事件一致性、部件缺失、关节间隙和时间平滑性分别报告，不用单个指标替代全部检查。

若 C 额外使用了 B 没有的网格、绑定或运动初始化，应单独列出，并补做“相同结构约束与先验、仅移除事件监督”的对照，才能识别事件本身的贡献。现有图片保留真实候选标签，不将内部实验截图重新命名为尚未完成的三组对照。

> 另有一条**严格输入支线**：仅使用修正 G0、原始事件流和相机内外参，不使用视频、Blender 或旧运动。它与上面的融合版不是同一流程；2026-10-18 的 `v5-plane` 仍为 0 / 266 个事件窗口通过，未替换网页。入口和完整失败记录保留在文末的“独立探索”中。

## 视频帧 + 事件流：现有融合实现

> 历史网页记录（2026-10-16，非本轮重建）：网页保留 `lego-global-detail-v1`，399 个事件区间中通过 382 个，仍有 17 个区间未通过，尚未通过全序列严格验收。`lego-roof-back-detail-v4` 因新增局部内部缺口被拒绝，未替换网页模型。播放已撤销“完整缓存后播放”，恢复短缓冲；流畅性问题尚未完全解决。

### 主要功能

- 读取单目 RGB 视频及原始 `x, y, t, p` 事件流，自动校准事件时间偏移与正负对比度阈值。
- 使用稀疏控制、刚体分组与硬关节前向运动学描述动态高斯运动。
- 仅用训练事件区间优化平滑时序校正，并对全部 399 个区间重新渲染审计。
- 在冻结几何和运动的前提下，对重建刚性表面的颜色、尺度和不透明度进行局部细化。
- 使用 Blender 三角网格作为明确标注的几何先验，补全观测相机不可见的背面与侧面。
- 在服务器 GPU 上实时渲染动态高斯，通过本地浏览器完成播放、逐帧检查和自由视角观察。
- 保存输入、模型、校正参数及审计结果的 SHA-256，避免不同实验产物被误配。

### 处理流程

1. **事件监督构建**：从 RGB 视频与原始事件流生成统一时间轴、事件计数和固定训练/留出划分。
2. **动态初始化**：从已有高斯、Lego 部件及相机参数生成 400 帧基础运动缓存。
3. **运动与关节校正**：利用原始事件变化、前景覆盖和接触约束优化平滑刚体运动。
4. **表面与外观细化**：修复可见刚性表面，并只调整新增表面的外观属性。
5. **多视角先验补全**：加入仅在轨道视角可见、在观测相机中不可见的网格先验采样点。
6. **独立审计与浏览**：执行事件一致性、部件完整性、多视角覆盖和序列化一致性检查。

## 已有融合实验结果（2026-10-14）

以下数字来自本地保存的 `lego-detail-v3` 与 `lego-multiview-v2` 审计记录，属于融合流程内部版本比较，不是原始 SC-GS、仅视频帧、视频帧加事件的三组消融，也不代表当前网页候选。所有事件指标均由实际高斯重新渲染得到，不使用视频帧替代渲染结果。

### 事件一致性

| 指标 | 细节增强前 v1 | 历史 v3 |
| --- | ---: | ---: |
| 全序列通过区间 | 377 / 399 | **380 / 399** |
| 平均事件余弦相似度 | 0.938474 | **0.943375** |
| 平均极性一致率 | — | **0.936914** |
| 平均相对 L1 | — | **0.380272** |
| 无事件支持变化比例 | — | **1.3307%** |
| 留出区间通过数 | 76 / 79 | **76 / 79** |
| 留出区间事件余弦 | — | **0.935391** |

该 v3 候选仍有 19 个区间未通过：余弦、极性、相对 L1 和无事件支持变化等失败原因可能重叠。因此其 `approved_for_export` 为 `false`，只允许显式导出带警告的候选预览。

### 图像细节与部件完整性

外观 v3 仅优化 88,878 个重建刚性表面高斯的灰度颜色、三轴尺度和不透明度；全部 XYZ、四元数、绑定、关节角及 400 帧运动保持不变。

| 参考相机指标 | v1 | v3 |
| --- | ---: | ---: |
| 铲臂区域 RGB L1 | 0.025700 | **0.024803** |
| 铲臂区域 Sobel 幅值误差 | 0.012448 | **0.011962** |
| 铲臂区域边缘 F1 | 0.877785 | **0.883694** |
| 可见部件内部缺失像素 | 81 | **56** |
| 缺失超过 10% 的部件帧 | 0 | **0** |
| 最大缺失连通块 | 2 px | **2 px** |

下图从左到右依次为 RGB 参考、内部 v1 候选和内部 v3 候选；两张图均不是 A / B / C 三阶段对照，不将其中的 v1 标为“仅视频帧”。

<p align="center">
  <img src="assets/experiments/detail-full-f100.png" width="100%" alt="第 100 帧全图细节对比">
</p>

<p align="center">
  <img src="assets/experiments/detail-arm-f100.png" width="100%" alt="第 100 帧铲臂局部细节对比">
</p>

### 多视角几何先验补全

`lego-multiview-v2` 在紧凑父模型上加入 316,916 个轨道视角可见的先验高斯，最终包含 1,847,203 个高斯。这里的背面和侧面来自给定 Blender 网格与材质先验，不是由单目观测恢复出的未知真实几何。

下表使用 alpha 阈值 0.5，统计运动刚体背面视角的前景缺失率：

| 覆盖指标 | 补全前 | 补全后 |
| --- | ---: | ---: |
| 主采样视角：像素加权缺失率 | 18.0649% | **0.3445%** |
| 主采样视角：部件等权缺失率 | 7.7211% | **0.2608%** |
| 主采样视角：最大缺失连通块 | 3,098 px | **77 px** |
| 留出视角：像素加权缺失率 | 12.9274% | **0.3481%** |
| 留出视角：部件等权缺失率 | 6.5821% | **0.2888%** |
| 留出视角：最大缺失连通块 | 3,044 px | **43 px** |

第 100 帧多视角检查：

<p align="center">
  <img src="assets/experiments/multiview-f100.png" width="100%" alt="第 100 帧多视角渲染">
</p>

独立留出时间与视角检查：

<p align="center">
  <img src="assets/experiments/multiview-heldout.png" width="100%" alt="多视角留出检查">
</p>

### 实时浏览器预览

浏览器显示的是服务器端 CUDA 渲染的动态三维高斯，不是输入 MP4 或以视频替代三维重建。双视窗共享时间轴、相机和背景，对比清单指定的未校正初始动画与候选。该初始动画不等同于一次受控的原始 SC-GS 基线重跑。

播放已恢复为少量帧预缓冲后开播，边接收边播放，不再要求浏览器先收齐当前视角的全部 400 帧。服务器已有的正视预热与逐帧缓存仍保留；视角、背景或清晰度改变时按需重新渲染，不修改模型、运动或默认 JPEG 画质。实际流畅度仍取决于渲染、传输与浏览器解码，撤销全量等待不等于任意连接下均能保证 60 FPS。

所有常驻文字和按钮仍位于画布外，播放状态在时间轴上方显示，事件未通过警告继续保留。已撤回的完整缓存实验及其验收记录留作历史资料，不代表当前播放策略。

<p align="center">
  <img src="assets/experiments/viewer-f100-yaw35.jpg" width="100%" alt="第 100 帧自由视角双视窗预览">
</p>

## 环境安装

推荐环境：Python 3.10、PyTorch 2.1.2、CUDA 12.1，以及支持 CUDA 的 NVIDIA GPU。

```bash
git clone --recursive https://github.com/SuleynanAuir/MonoEvent-RT3D-DMR.git
cd MonoEvent-RT3D-DMR
```

AutoDL 环境可直接运行：

```bash
bash scripts/setup_autodl.sh
source scripts/autodl_env.sh
```

手动安装时，先安装与本机 CUDA 匹配的 PyTorch，再安装项目依赖和两个 CUDA 子模块：

```bash
pip install -r requirements.txt
pip install -r requirements-event.txt
pip install -r requirements-viewer.txt
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

## 数据目录

大文件数据不会提交到 Git。以下目录对应视频帧 / 事件融合流程，包含其实际使用的额外几何与初始化先验；不是严格输入支线的必需数据清单：

```text
Lego_data/
├── Lego.mkv
├── Lego_event
├── Lego.blend
├── camera_intrinsics.json
├── fixed_camera.json
└── fan_contrastive_g0/
    └── contrastive_g0.ply
```

仓库默认忽略 `Lego_data/`、`outputs/`、PLY、模型权重和 NumPy 大数组。请将实验产物写入 `outputs/` 或独立数据盘。

## 复现现有融合流程

以下命令运行现有的含结构先验融合实现，不是原始 SC-GS 或纯视频帧消融入口。严格输入流程请使用文末独立入口，不要运行这里读取视频、部件或 Blender 派生数据的命令。

### 1. 构建事件监督

```bash
python scripts/prepare_lego_events.py \
  --data /path/to/Lego_data \
  --output outputs/lego-event-supervision
```

脚本会固定划分训练与留出区间；每五个事件区间留出一个，不参与梯度或检查点选择。

### 2. 运行硬关节事件校正

以下命令中的基础缓存、部件、绑定和关节报告由对应的 `cache_lego_initial_motion.py`、部件导出/绑定脚本及关节诊断脚本生成。输出目录必须不存在。

```bash
python scripts/refine_lego_temporal.py \
  --baseline outputs/lego-baseline \
  --supervision outputs/lego-event-supervision/supervision.npz \
  --parts outputs/lego-init/blender-surface-parts.npz \
  --binding outputs/lego-binding/initial-binding.npz \
  --joint-report outputs/lego-joints/joint-report.json \
  --foreground-weight 12 \
  --lr 0.01 \
  --iterations 800 \
  --output outputs/lego-event-hard-joint
```

审计未通过时脚本保留全部诊断结果，但不会自动把候选标记成可正式导出。

### 3. 导出研究候选并启动浏览器

```bash
python scripts/export_lego_event_preview.py \
  --baseline outputs/lego-baseline \
  --candidate outputs/lego-event-hard-joint \
  --output outputs/lego-event-preview \
  --allow-unapproved-preview

python viewer.py \
  --sequence_path outputs/lego-event-preview/lego_event_candidate.npz \
  --host 127.0.0.1 \
  --port 8892 \
  --cache auto
```

浏览器打开 `http://127.0.0.1:8892/?autoplay=1&loop=1`。远程 GPU 建议通过 SSH 隧道访问，不要把查看器直接暴露到公网。

#### 可选 Mesh 展示（从高斯重建）

查看器的「渲染设置 → 显示模式」支持 `Gaussian` 和 `Mesh`，默认保留高斯对比。
Mesh 模式保持双视窗：左侧为 V1 Mesh（原始 GS），右侧为 V3 Mesh（最新优化 GS），两边同帧、同相机、同尺寸。此处 V1/V3 指查看器原有左右两份结果，并非历史实验目录的版本号。点击「放大对比」可将两个视窗一起全屏显示。
两份 Mesh 分别来自各自高斯的多视角 RGB/深度 TSDF 融合，不读取原始三角网格。V1 仅复用原有部件绑定标签，并从自身逐帧高斯恢复刚体变换、检查全部帧的刚性误差；V3 沿用自身事件校正运动。两份运动独立存入网格，避免将 V3 的运动套用给 V1。
细节版默认使用 1280 像素、90 个视角、0.0015 体素、0.65 透明度阈值和 4 体素截断距离；用受原始表面邻域限制的 Screened Poisson 重建改善细碎裂缝，并根据多视角高斯的空域证据剔除不受支持的表面，减少误填真实孔洞。颜色从可见的高分辨率高斯视角单独烘焙，显示时不再叠加光照。可使用 `--surface-method tsdf` 对照原始提取方法。
当整体表面正则化降低细节相似度时，可用 `scripts/refine_viewer_mesh_gaps.py --base <同版本细节网格> --surface <同版本经空域筛查的连续网格> --output <新目录>` 仅添加边界附近的补片，完整保留细节网格原有顶点、三角面和颜色。两种网格必须来自同一份高斯，不能跨 V1/V3 混用。
表面仍有细碎凸起时，可对左右两份网格分别运行 `python scripts/denoise_viewer_mesh.py --input <mesh.npz> --output <新目录>`，再将 manifest 指向处理后的文件。默认沿已有三角面连接进行 16 轮受限 Taubin 平滑，边界顶点最多移动 1 个体素，内部最多 2 个体素；不增删三角面、不跨部件或跨孔洞连接，并完整保留各自的运动。按实际保存的 float32 精度检查三角面朝向，对可能翻折的顶点局部回退。另做 3 轮保留颜色边界的顶点颜色去噪，输出报告记录每个部件的位移及局部起伏变化。查看器以宽高各 2 倍采样后缩回输出尺寸，改善微小三角面和轮廓的抗锯齿，画面大小和相机保持不变。
需要进一步减少毛躁时使用 `--preset smooth`：本次处理采用 64 轮几何平滑和 10 轮颜色去噪，保持相同的位移限制及防翻折检查。左右两版使用相同参数，输出保留前次去噪报告与输入文件哈希；建议对当前结果运行一次后检查轮廓与孔洞，不要无检查地反复叠加。此预设偏重平顺度，细小纹理的对比度可能略降。

需要增加三角面精细度时，可分别执行 `python scripts/subdivide_viewer_mesh.py --input <已去噪的 mesh.npz> --output <新目录>`。默认对超过 4/3 个体素的长边做一次自适应细分，每个部件的三角面最多增加至原来的 2 倍；相邻面共用边中点，过渡面同步拆分以避免裂缝。原有顶点和颜色保持不变，新增顶点按去噪后表面的切平面估计曲率，偏离线性中点最多 0.15 个体素且不超过边长的 8%；边界与锐边附近只做线性细分。原网格中的极薄面与重复面保持原样，按实际 float32 输出检查子面朝向，不安全区域回退或保持原面。左右独立处理并保留各自全部运动帧，新增顶点颜色沿边插值。细分改善曲面的离散精度，不会凭空恢复高斯源中缺失的真实结构或纹理；新增面数也会增加显存与渲染开销，切换 manifest 前应检查多视角效果和播放速度。
低置信度表面、小碎片及弹性带会被略去，网格是高斯表面的近似，不代表重建精度或事件验收有所提升。

细节颜色可在去噪后再拟合一次。`scripts/capture_viewer_mesh_views.cjs` 从正在运行的查看器采集 24 组训练视角和 6 组验证视角，每张图包含左右各自的高斯结果；拟合时严格按侧别裁切。`scripts/refine_viewer_mesh_colors.py` 仅优化顶点颜色，保持几何、三角面、部件归属及全部运动参数不变，限制 RGB 改变量，并通过随机表面邻域约束及递减学习率减少颗粒和单视角过拟合。已有颜色均匀区域的改动上限降至全局上限的 5%，避免白色平面新增斑点；沿已有颜色边缘逐步放宽限制。验证视角不参与梯度计算，候选应通过新视角对照后再部署；可以用采集脚本的 `check` 模式额外采集 4 组检查视角。颜色拟合不能补回缺失的几何或证明真实重建精度提高。

```bash
node scripts/capture_viewer_mesh_views.cjs http://127.0.0.1:8892 /path/to/color-views
python scripts/refine_viewer_mesh_colors.py \
  --comparison /path/to/comparison-manifest.json \
  --input /path/to/v1-denoised/mesh.npz --views /path/to/color-views \
  --output /path/to/v1-color-refined
# 对 V3 输入单独执行同一命令；不能将 V1 的颜色套给 V3。
node scripts/capture_viewer_mesh_views.cjs http://127.0.0.1:8892 /path/to/color-check check
```

在现有 CUDA 环境安装可选依赖并生成一次网格：

```bash
pip install --no-deps --no-build-isolation -r requirements-mesh.txt
pip install open3d-cpu==0.19.0 'numpy<2'
python scripts/reconstruct_viewer_mesh.py \
  --comparison /path/to/comparison-manifest.json \
  --side left --left-binding /path/to/original/initial-binding.npz \
  --output /path/to/v1-gaussian-mesh
python scripts/reconstruct_viewer_mesh.py \
  --comparison /path/to/comparison-manifest.json \
  --side right --output /path/to/v3-gaussian-mesh
```

在对比 manifest 中增加 `"mesh_paths": {"left": "/path/to/v1-gaussian-mesh/mesh.npz", "right": "/path/to/v3-gaussian-mesh/mesh.npz"}`，重启查看器即可选择 Mesh。两份文件必须与各自高斯、运动和侧别匹配；缺失或混用时禁用 Mesh 模式，避免误标。旧的单份 `mesh_path` 配置需要重新生成双份网格后迁移。
也可通过 `http://127.0.0.1:8892/?mode=mesh&autoplay=1&loop=1` 直接打开。
切换模式保留时间与相机；Mesh 加载失败时高斯仍可使用。重新生成高斯或校正参数后，需要重新生成匹配的网格。

### 4. 测试

```bash
python -m unittest discover -s tests -v
node --test tests/test_viewer_playback.cjs
```

浏览器端到端测试位于 `scripts/test_gaussian_viewer_ui.cjs`，需要额外安装 `playwright-core` 并配置 Chrome 路径。

### 5. 车顶与固定背侧的受保护表面细化

此流程为实验代码：本轮 `lego-roof-back-detail-v4` 未通过缺口验收，未部署；保留代码不代表该候选已达标。

此阶段叠加在已审计的颜色残差候选上，不改变动态部件绑定或运动：

- `prepare_lego_back_detail.py` 在已知三角网格上采样薄层各向异性高斯；颜色使用同部件先验，不生成未知背面纹理。
- `preview_lego_roof_detail.py` 对完整 400 帧检查新增和替换点的观测相机贡献，并迭代剔除超限点。出现新的目标部件内部缺口时，恢复原始高斯及邻域支撑，不使用二维补洞。
- `package_lego_surface_detail.py` 仅打包已收敛结果，保留父模型、颜色与运动数据，输出独立的 `model/surface-detail.npz` 与带哈希的报告；不会自动部署。
- `render_audit_lego_completion.py` 和 `render_lego_multiview.py` 从打包结果重新渲染。最终运行 `check_lego_surface_detail_release.py`，检查整段 RGB / 事件无回退、序列化一致性和多视角新增内部缺口。

36 个构造视角及用于缺口修复的额外 24 个视角均属于选择数据，不能称为独立验证。启用这 24 个修复视角后，发布检查还要求 `export_lego_multiview_masks.py --detail-check` 生成的另一组 12 个未参与选点的视角；相机记录不能重叠。

运行各脚本的 `--help` 查看所需路径。发布检查示例（各目录须已完成实际渲染）：

```bash
python scripts/check_lego_surface_detail_release.py \
  --parent /path/to/audited-parent \
  --candidate /path/to/packaged-detail \
  --supervision /path/to/supervision.npz \
  --masks /path/to/rigid-masks-36.npz \
  --heldout-masks /path/to/rigid-masks-heldout-24.npz \
  --detail-masks /path/to/roof-back-check-12.npz \
  --output /path/to/packaged-detail/qa/release-check.json
```

浏览时使用 `viewer.py --comparison_path /path/to/manifest.json`，其中 `right_baseline` 和 `candidate` 分别指向新包内的 `baseline/` 与 `candidate/`。**仅打开父模型 PLY 不包含这一细化层**；旧单序列导出入口会拒绝此类候选，避免静默丢失新增高斯。

有限角度零新增内部缺口不等于任意视角无缺陷，结构边缘与网格的一致性也不等于真实背面纹理准确。事件尚未整段通过时，UI 必须继续显示未通过警告。

## 代码结构

```text
gaussian_renderer/   CUDA 高斯渲染入口
scene/               相机、数据读取、Gaussian 模型与形变模型
utils/               事件监督、运动、关节、网格和序列工具
scripts/             Lego 实验、诊断、审计、打包与多视角流程
viewer/              浏览器前端
viewer.py            GPU 渲染服务
tests/               Python 与 JavaScript 回归测试
```

## 独立探索：仅 G0 + 事件 + 相机

这条支线的输入限制与“视频帧 + 事件流”融合版不同：只接受修正 G0、原始事件流、相机内参和外参。历史融合版的结果不作为这条支线的输入或成绩；最近完整候选仍未通过验收。

<details>
<summary>展开严格输入流程的运行命令、实验记录和未解决问题</summary>

### 输入与独立重建入口

2026-10-18 新增事件-only 流程，**与上文依赖 RGB/Blender 的融合流程分开**。它只读取显式给出的修正 G0、原始事件流、相机内参与外参，不读取视频、网格、动画、旧绑定、旧运动或旧监督。原始事件可以是没有扩展名的 NPY 数组（例如 `Lego_event`），不能用事件可视化视频替代。

运动默认从恒等状态开始学习：仅依据 G0 位置建立变形图，固定首帧、颜色、尺度、透明度和高斯行序，优化连续时间的节点平移与旋转。原生分辨率渲染后在对数亮度域比较事件；局部刚性和时间平滑是正则约束，不是输入的真实部件或动画。事件积分产生的训练目标由 G0 和事件派生，不是额外 RGB 观测。

可在 `--fit-split all` 下增加 `--event-flow-initialization`：仅由 G0 渲染和累计事件构造代理图，使用无预训练的经典光流生成初值。初始深度用于反投影，是先验而非真实动态深度；隐藏节点、低纹理及不一致流会被过滤。每段以实际高斯事件损失比较多个初值幅度，包含不采用光流的候选。联合优化先逐一监督最短事件窗，再采样多窗跨度。

若 G0 的黑底渲染与事件传感器亮度模型不匹配，可显式使用 `--estimate-event-background --log-floor .0001`。程序仅依据初始亮前景和累计负事件推估一个全局、固定背景参数 B，并统一使用 `log(gray + (1-alpha)*B)` 生成目标、训练与审计；不会用逐像素偏置掩盖运动误差。该估计与未知事件阈值耦合，不等于真实背景标定，证据不足会停止并保留诊断。输出预览仍可使用黑底。

可选 `--propagate-event-flow` 将可信节点初值沿有界 G0 三维邻接传播到被遮挡或低纹理节点，使用局部刚体拟合与 ARAP，静态锚点仍由事件活动的保守假设确定。这不是原有部件绑定，也不证明背面运动正确；近邻跨部件和历史观测滞后仍需通过真实高斯渲染检查。

另一条实验分支是 `--rigid-feature-initialization --feature-reference-reacquisition`：在原 G0 的不透明、深度一致表面提取特征，用事件积分代理跟踪，再跨时间推断刚体组。每个已绑定高斯只受一个刚体变换控制，位置与四元数同时旋转，不再用多个部件的加权形变拉伸单个部件。未知绑定保留为单独的恒等初始化先验，不算已重建；刚体分组检查也不代替原生事件审计。该分支不能与 `--event-flow-initialization` 同用。

可选 `--geodesic-binding` 把分组种子沿 G0 占用体素的有界互近邻路径传播；`--binding-cell-fraction`、`--binding-radius-fraction`、`--binding-path-fraction` 分别限制体素、欧式距离及路径长度（相对于 G0 包围盒对角线）。冲突、无路径及超预算仍未知，不自动扩半径或降级为最近邻。它只是空间先验，不能证明隐藏表面属于同一运动部件。

`--feature-method sift` 是可选的原 G0 双向描述子匹配分支，不读取预训练模型，也不把断链补成连续观测。它在当前 Lego 实测中未比角点跟踪提供更多有效运动支持，默认仍为 `corners`。

`--motion-model camera_plane` 在刚体分支中显式假设每个 G0 高斯的相机深度固定，只允许绕相机光轴旋转与成像平面内平移；初始化及训练全程保持此约束。它可减少单目姿态歧义，但会排除真实的面外运动，因此不是物理规律保证。默认仍是未限制的 `se3`，所有原生事件门槛不变。

```bash
python scripts/reconstruct_lego_event_only.py \
  --g0 /path/to/corrected_g0.ply \
  --events /path/to/Lego_event \
  --intrinsics /path/to/camera_intrinsics.json \
  --extrinsics /path/to/fixed_camera.json \
  --g0-time 0 --time-scale 1 \
  --fit-split all --nodes 256 --warmup-steps 32 --polish-steps 600 \
  --output /path/to/new-event-only-run

python scripts/render_event_only_sequence.py \
  --g0 /path/to/new-event-only-run/gaussians.ply \
  --checkpoint /path/to/new-event-only-run/deform.pth \
  --fps 30 --width 640 --output /path/to/new-preview
```

注意：示例的 G0 对应 `t=0`、事件时间单位为秒，以及默认正/负阈值 `0.2` 都需要实际确认；程序不会从旧视频标定中继承这些数值。当前相机入口明确支持 boundary 像素中心约定与转置保存的 3DGS `R`/`T`，不猜测其他约定，不跟随相机 JSON 内的历史文件引用。

输出包含原样复制的 `gaussians.ply`、自定义变形图 `deform.pth`、输入 SHA-256、训练记录和整段 `event-audit.json`。`deform.pth` **不是原版 SC-GS 网络权重**，须配合 `utils/event_only_checkpoint.py` 和 `utils/event_only_motion.py` 加载。导出器生成 `G0.ply`、`deform.pth`、预览视频与关键帧；可用 `--write-ply-times 0,6.65` 额外导出少量真正变形后的 PLY，不批量复制全序列百万点文件。

`--fit-split all` 使用全部事件恢复整段运动，审计只能称为训练数据上的一致性检查，不能称为独立验证。若使用留出窗口，应选择 `--fit-split train --warmup-steps 0`；训练目标不跨越留出窗口，也不构建含留出事件的累计目标。未知触发阈值、未确认的 G0/时间对应关系和失败区间都会保留为验收阻碍。仅仅成功训练、保存或播放，不代表整段事件通过或单目不可见部分真实正确。

`scripts/diagnose_event_only_run.py --run-dir /path/to/run` 可汇总实际优化覆盖、原生分辨率指标和未通过原因；缺少最终报告不会视为通过。`scripts/audit_event_only_checkpoint.py --checkpoint /path/to/run/deform.pth --g0 /path/to/run/gaussians.ply --events /path/to/Lego_event --output /path/to/new-consistency.json` 会重载模型并整段重渲染，补充时间累计诊断。后者保留原生严格门槛，额外报告理想事件模型的必要残差界 `C+ + C-`；不超界并不足以证明运动正确，正负抵消也不等于没有运动。

新流程测试：`python -m unittest discover -s tests -p 'test_event_only*.py'`。本流程不使用上文融合实验中的模型、轨迹或标定结果。

#### 前一阶段实测状态（2026-10-18，尚未成功）

三次完成的实验都优化了完整 6.65 秒、266 个原生事件窗口，保留全部 1,841,816 个 G0 高斯。下表是全部事件参与训练后的拟合一致性，不是独立验证；原生严格门槛没有调整。

| 候选 | 平均事件余弦 | 平均相对 L1 | 通过窗口 |
| --- | ---: | ---: | ---: |
| v1：恒等初值 | 0.0896 | 4.7414 | 0 / 266 |
| v3：事件代理初值和背景估计 | 0.4240 | 3.4259 | 0 / 266 |
| v4：增加有界邻接传播 | 0.4811 | 3.0838 | 0 / 266 |

v4 序列化重载后重新渲染的指标与训练审计一致，但铲斗仍有明显非刚性拉伸，因此模型与预览均是**失败的实验候选**，没有替换网页。三维刚体发现诊断也未找到具有足够跨时证据的动态组；静态组的 identity 不能算动态运动重建。

实测后还发现旧光流初值允许在中断后从过期坐标继续采样相邻帧光流，不能保证 G0 点身份连续。现已改为断链后冻结未验证先验、不自动恢复观测身份，196 项云端测试通过。修复后的真实数据诊断从 72 个初始可见节点中保留 11 条全程连续轨迹，拦截 829 次缺少 G0 连续身份的局部候选，仍未建立足够的动态刚体组。代码测试通过不等于重建通过。

旧候选保留原始执行源码和失败记录，不因后续代码修复而改标为已修复或通过；v3/v4 源码清单已逐项校验归档，v1 缺完整执行源码哈希清单。上述实验运行时，G0 文件身份及 t=0 对齐尚未得到用户确认，历史记录不追溯改写。

#### 用户确认后的续优化

用户现已确认 `fan_contrastive_g0/contrastive_g0.ply` 是修正后的 G0，且对应事件 `t=0`。新运行显式记录 `--identity-time-confirmed` 和四个输入哈希；事件正负阈值 `0.2` 仍未独立标定。继续仅使用该 G0、原始事件流和相机内外参，不使用旧视频、Blender、部件标签、绑定或运动权重。

`scripts/diagnose_event_feature_groups.py` 可先执行新特征/分组/绑定初始化诊断，接受同样的四个显式原始文件；加 `--make-binding` 输出 `initialization-only.pth`。它没有训练，不能当成最终 `deform.pth` 或通过验收的模型。派生代理图、跟踪数组和源码快照均仅用于同输入审计，不是新增观测模态。

诊断入口的 `--score-binding-photometry` 对 G0 可见表面的小块进行跨时光度评分。候选使用共同的有效时刻、原 G0 的绝对对数亮度和同一事件积分模型；目标可见性未被测量，低信息、相互冲突及高残差保留未知。该诊断仅输出建议标签和证据，不修改绑定，更不把背面当作已观测。

若所有运动组共同有效的时刻过少，可用 `--score-binding-photometry-pairs` 分别比较恒等先验与每个动态候选。不同二元比较的有效时刻可能不同，因此分数不能跨比较排序，也不能直接合并成互斥部件标签；还需要动态组之间的竞争、几何传播及整段高斯渲染检查。当前训练不自动采用这些诊断标签。

本轮 `v5-plane` 已完成全部 6.65 秒的 67 个非零时间节点逐段拟合（每节点 32 步）及 600 步联合优化，266 个原生窗口均单独参与优化。输入 G0 的 1,841,816 行及外观属性全部保留；角点分支使用 904 个初始可见特征，推断 4 个动态候选组，但这些组不是已知部件标签。绑定未知处仍是冻结先验，不能当成已恢复运动。

| 全序列候选 | 平均事件余弦（越高越好） | 平均相对 L1（越低越好） | 原生通过窗口 |
| --- | ---: | ---: | ---: |
| 前阶段 v4 | 0.4811 | 3.0838 | 0 / 266 |
| 本轮 v5-plane | 0.2261 | 2.5147 | 0 / 266 |

**v5-plane 被拒绝，没有替换网页。** 相对 L1 降低不代表整体更好：事件余弦明显下降，平均极性一致率由 0.7249 降至 0.4918，无事件支持变化比例由 0.5970 升至 0.6300。两次实验使用相同四个原始输入哈希；该表是候选之间的同输入拟合检查，不是控制变量消融或独立验证。

最终 `deform.pth` 已重载并重新渲染全部 267 个事件端点；266 个窗口的四项原生指标与训练审计逐项完全一致。201 帧、30 FPS、640×640 的预览已导出，但属于失败候选预览，不是已验收的动态模型。

只读诊断发现，在大幅运动的约 1.5–2.0 秒，上臂故障位置附近的轨迹支持不足。例如原像素 `(265,155)` 半径 20 像素内的 22 个特征，在 1.7 秒全部失去有效跟踪；半径 50 像素内也只剩 7 条，未达到组拟合所需 8 条。这里的区域仅用于故障定位，不作为输入分割；没有有效轨迹不等于没有事件。缺失组姿态的插值跨过了运动峰值，后续优化未解决残影。

另记录了一项待实验验证的监督风险：刚体初值的 23 个超出重投影门槛的点次仍进入特征先验，另有 572 个点次缺少当时的组级姿态支持；后者不能直接认定为错误轨迹。尚未证明它是残影主因，本轮模型也没有追溯改标为已修复。当前云端 356 项回归测试通过，实际执行的 16 个源码文件与本地及归档哈希一致；代码测试通过不等于重建验收通过。

</details>

## 实验边界

- 当前结果是单场景研究候选，不能直接代表其他场景的泛化性能。
- 融合流程的留出事件不参与梯度和检查点选择，但相邻 RGB 帧仍可能参与外观或覆盖约束，因此它不是全新场景测试集。严格输入支线的全事件拟合则只报告训练内一致性。
- 多视角补全评估的是给定网格先验下的投影覆盖，不证明不可见面的真实形状、材质或物理正确性。
- 刚体、关节几何检查不等于无视觉缺口或动力学验证；事件一致性、部件覆盖和运动学约束应分别验收。
- 弹性带没有使用刚性网格补面；透明材质也未由不透明三角掩码完整建模。
- 所有失败实验和大体积审计结果保留在本地 `outputs/`，不上传到 GitHub。

## 致谢

本项目基于以下工作和代码继续开发：

- [SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes](https://github.com/yihua7/SC-GS)
- [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

使用或扩展本仓库时，请同时遵守原项目许可并引用相关工作：

```bibtex
@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}
```

## 许可证

项目采用 [MIT License](LICENSE)。
