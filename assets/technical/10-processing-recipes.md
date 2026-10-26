# 补充处理与历史命令

本页保留 Mesh、混合展示与受保护表面细化的已有命令。输入路径是模板；各入口依赖已验证的对应资产，不能视为按顺序复制即可从原始数据生成全部结果的一键流程。历史 V1/V3 标签不改变其来源。

#### 可选 Mesh 展示（从高斯重建）

查看器的「渲染设置 → 显示模式」支持 `Gaussian` 和 `Mesh`，默认保留高斯对比。
Mesh 模式保持双视窗：左侧为 V1 Mesh（原始 GS），右侧为 V3 Mesh（最新优化 GS），两边同帧、同相机、同尺寸。此处 V1/V3 指查看器原有左右两份结果，并非历史实验目录的版本号。点击「放大对比」可将两个视窗一起全屏显示。
两份 Mesh 分别来自各自高斯的多视角 RGB/深度 TSDF 融合，不读取原始三角网格。V1 仅复用原有部件绑定标签，并从自身逐帧高斯恢复刚体变换、检查全部帧的刚性误差；V3 沿用自身事件校正运动。两份运动独立存入网格，避免将 V3 的运动套用给 V1。
细节版默认使用 1280 像素、90 个视角、0.0015 体素、0.65 透明度阈值和 4 体素截断距离；用受原始表面邻域限制的 Screened Poisson 重建改善细碎裂缝，并根据多视角高斯的空域证据剔除不受支持的表面，减少误填真实孔洞。颜色从可见的高分辨率高斯视角单独烘焙，显示时不再叠加光照。可使用 `--surface-method tsdf` 对照原始提取方法。
当整体表面正则化降低细节相似度时，可用 `scripts/mesh/refine_viewer_mesh_gaps.py --base <同版本细节网格> --surface <同版本经空域筛查的连续网格> --output <新目录>` 仅添加边界附近的补片，完整保留细节网格原有顶点、三角面和颜色。两种网格必须来自同一份高斯，不能跨 V1/V3 混用。
表面仍有细碎凸起时，可对左右两份网格分别运行 `python scripts/mesh/denoise_viewer_mesh.py --input <mesh.npz> --output <新目录>`，再将 manifest 指向处理后的文件。默认沿已有三角面连接进行 16 轮受限 Taubin 平滑，边界顶点最多移动 1 个体素，内部最多 2 个体素；不增删三角面、不跨部件或跨孔洞连接，并完整保留各自的运动。按实际保存的 float32 精度检查三角面朝向，对可能翻折的顶点局部回退。另做 3 轮保留颜色边界的顶点颜色去噪，输出报告记录每个部件的位移及局部起伏变化。查看器以宽高各 2 倍采样后缩回输出尺寸，改善微小三角面和轮廓的抗锯齿，画面大小和相机保持不变。
需要进一步减少毛躁时使用 `--preset smooth`：本次处理采用 64 轮几何平滑和 10 轮颜色去噪，保持相同的位移限制及防翻折检查。左右两版使用相同参数，输出保留前次去噪报告与输入文件哈希；建议对当前结果运行一次后检查轮廓与孔洞，不要无检查地反复叠加。此预设偏重平顺度，细小纹理的对比度可能略降。

需要增加三角面精细度时，可分别执行 `python scripts/mesh/subdivide_viewer_mesh.py --input <已去噪的 mesh.npz> --output <新目录>`。默认对超过 4/3 个体素的长边做一次自适应细分，每个部件的三角面最多增加至原来的 2 倍；相邻面共用边中点，过渡面同步拆分以避免裂缝。原有顶点和颜色保持不变，新增顶点按去噪后表面的切平面估计曲率，偏离线性中点最多 0.15 个体素且不超过边长的 8%；边界与锐边附近只做线性细分。原网格中的极薄面与重复面保持原样，按实际 float32 输出检查子面朝向，不安全区域回退或保持原面。左右独立处理并保留各自全部运动帧，新增顶点颜色沿边插值。细分改善曲面的离散精度，不会凭空恢复高斯源中缺失的真实结构或纹理；新增面数也会增加显存与渲染开销，切换 manifest 前应检查多视角效果和播放速度。
低置信度表面、小碎片及弹性带会被略去，网格是高斯表面的近似，不代表重建精度或事件验收有所提升。

细节颜色可在去噪后再拟合一次。`scripts/viewer/capture_viewer_mesh_views.cjs` 从正在运行的查看器采集 24 组训练视角和 6 组验证视角，每张图包含左右各自的高斯结果；拟合时严格按侧别裁切。`scripts/mesh/refine_viewer_mesh_colors.py` 仅优化顶点颜色，保持几何、三角面、部件归属及全部运动参数不变，限制 RGB 改变量，并通过随机表面邻域约束及递减学习率减少颗粒和单视角过拟合。已有颜色均匀区域的改动上限降至全局上限的 5%，避免白色平面新增斑点；沿已有颜色边缘逐步放宽限制。验证视角不参与梯度计算，候选应通过新视角对照后再部署；可以用采集脚本的 `check` 模式额外采集 4 组检查视角。颜色拟合不能补回缺失的几何或证明真实重建精度提高。

```bash
node scripts/viewer/capture_viewer_mesh_views.cjs http://127.0.0.1:8892 /path/to/color-views
python scripts/mesh/refine_viewer_mesh_colors.py \
  --comparison /path/to/comparison-manifest.json \
  --input /path/to/v1-denoised/mesh.npz --views /path/to/color-views \
  --output /path/to/v1-color-refined
# 对 V3 输入单独执行同一命令；不能将 V1 的颜色套给 V3。
node scripts/viewer/capture_viewer_mesh_views.cjs http://127.0.0.1:8892 /path/to/color-check check
```

在现有 CUDA 环境安装可选依赖并生成一次网格：

```bash
pip install --no-deps --no-build-isolation -r requirements/mesh.txt
pip install open3d-cpu==0.19.0 'numpy<2'
python scripts/mesh/reconstruct_viewer_mesh.py \
  --comparison /path/to/comparison-manifest.json \
  --side left --left-binding /path/to/original/initial-binding.npz \
  --output /path/to/v1-gaussian-mesh
python scripts/mesh/reconstruct_viewer_mesh.py \
  --comparison /path/to/comparison-manifest.json \
  --side right --output /path/to/v3-gaussian-mesh
```

在对比 manifest 中增加 `"mesh_paths": {"left": "/path/to/v1-gaussian-mesh/mesh.npz", "right": "/path/to/v3-gaussian-mesh/mesh.npz"}`，重启查看器即可选择 Mesh。两份文件必须与各自高斯、运动和侧别匹配；缺失或混用时禁用 Mesh 模式，避免误标。旧的单份 `mesh_path` 配置需要重新生成双份网格后迁移。
也可通过 `http://127.0.0.1:8892/?mode=mesh&autoplay=1&loop=1` 直接打开。
切换模式保留时间与相机；Mesh 加载失败时高斯仍可使用。重新生成高斯或校正参数后，需要重新生成匹配的网格。

#### 白模与表面线框（Clay）

显示模式中选择“V1 / V3 白模 · 清晰柔光”，或打开 `http://127.0.0.1:8892/?scene=lego&mode=clay`（风扇为 `scene=fan`）。默认使用纯白模；“显示表面线框”可开启低对比度线条，链接中可设置 `clay_style=wire`。已移除旧“灰模 · 网格”入口及对应资产加载，`clay_mesh` 不再是有效显示模式。

白模直接使用各侧原生 GS，保留覆盖率、轮廓、运动及 V3 局部 MLP 辅助变形。照明仅保留低频的柔和形体明暗，移除容易放大深度噪声的微小高光及局部遮蔽。亮度先按四个方向的连续性识别孤立亮暗颗粒，仅在不透明且深度连续的内部区域抑制噪点，再进行中值引导、深度约束的保边去噪。连续细线、薄轮廓与原始覆盖率保持；9 / 31 像素的分尺度明暗增强部件层次，并限制细碎纹理的增益，避免整体锐化放大毛躁。默认每侧 1024 px，保留 V1 / V3 双视图、同步相机、播放和截图；线框与纯白模分别缓存并预热。Lego 白模缓存预算为 384 MiB，以保留两套完整动画，避免第二套预热挤出第一套。

线框是用于展示的物体坐标网格，并非恢复出的四边形 Mesh 拓扑。每个高斯携带固定的规范帧材质坐标，在当前运动与遮挡下另渲染坐标场，再按表面方向混合三组平面网格。仅过滤线框坐标与光照引导，保留原 GS 的覆盖率和细节；细于像素的网格和不连续边界会淡出。线条随物体运动，不固定在屏幕上，也不使用跨帧混合。大角度视图中原 GS 的缺失表面仍可能显露。

`scripts/viewer/probe_gs_clay.py --scene lego --input <viewer.json> --output <新目录>` 保存不同帧／视角的原 GS 与白模线框对比及渲染字段。`tests/mesh/test_gs_clay.py` 检查覆盖率、遮挡、线条随材质坐标移动及远距离抗锯齿；`tests/geometry/test_gaussian_material.py` 用真实 CUDA 光栅器验证坐标覆盖颜色且不改变原始 SH 外观。线框开关不会改变运动帧、视角或 GS 模型参数。

以下离线网格修复工具保留用于几何实验；不参与当前白模显示，也不再通过 `clay_paths` 自动加载：

`scripts/mesh/regularize_clay_mesh.py --input <mesh.npz> --output <新目录>` 按运动组作局部定向采样、平面去噪和有距离限制的 Poisson 表面重建；`scripts/mesh/patch_clay_surface.py --input <去噪mesh.npz> --source <原mesh.npz> --camera <相机JSON> --output <新目录>` 从多个视角检测小封闭孔洞，仅在边界属于同一运动组、局部深度连续且靠近源表面时生成三维补片。补片随后随部件运动，而非每帧图像涂抹。依赖 Open3D 与 SciPy。修复是有尺度限制的几何推断，不是新的观测证据；细于修复尺度的孔洞可能被闭合，大面积未知表面仍不保证完整。

使用 `scripts/viewer/evaluate_clay_surface.py --before <原mesh.npz> --after <修复mesh.npz> --camera <相机JSON> --output <结果目录>` 生成相同相机、光照下的对比图、轮廓 IoU、内部高频粗糙度和小封闭背景区域统计。这些是显示效果代理，不能当作真实几何精度或事件监督指标。

完成补洞后，可运行 `scripts/mesh/refine_clay_features.py --input <补洞mesh.npz> --output <新目录> --radius .010 --max-move .004`（Lego；风扇使用 `.007` / `.0025`）。使用稳健局部平面拟合、静态／动态引导法线过滤和受限点到平面重建，保留原三角面及运动分组；反向薄片分开拟合，位移有上限，发生三角形翻转时回退。方法借鉴 [Static/Dynamic Filtering for Mesh Geometry](https://arxiv.org/abs/1712.03574) 的引导思想，采用局部点采样实现，并非该论文全局优化的复现。它不进行二维模糊、锐化或跨帧混合，不能从已有网格中恢复已经缺失的真实细节。保边资产默认以每侧 1024 px 显示；光照法线是去噪后的展示法线，不应视为原始几何微分法线或新的观测证据。

先补洞、后保边重建；补洞脚本拒绝已含规范法线的输入，防止新增顶点使用过期法线。`tests/mesh/test_clay_features.py` 验证平面降噪、直角保留、双面薄片厚度和位移限制，`tests/mesh/test_clay_shading.py` 验证法线随主运动与辅助关节旋转。

1024 px 灰模的自动播放以 30 FPS 开始，预缓冲约 0.6 秒，避免双侧高分辨率解码在启动时丢帧；图像质量与运动速度不变，也可手动选择 60 FPS。实际播放仍受客户端解码和网络速度影响。

#### 网页动态混合展示（Mesh＋Gaussian）

当目标是动态展示，单层网格加顶点颜色会损失原高斯的半透明边缘和细节，继续增面无法恢复这些外观信息。可额外启用明确标注的 `hybrid` 模式：每侧高斯绑定到该侧重建网格的三角面，保留重心坐标与规范帧中的偏移；网格自带的部件变换驱动高斯的位置与朝向，高斯继续绘制外观。这是混合表示，不是传统纯 Mesh，也不是 SuGaR 的训练实现。没有网格绑定的弹性部件沿用原高斯动画。静态偏移不被压缩，避免把体积外观拍扁或改变模型轮廓。

```bash
python scripts/mesh/bind_gaussians_to_viewer_mesh.py \
  --comparison /path/to/comparison-manifest.json \
  --left-binding /path/to/original/initial-binding.npz \
  --output /path/to/new-hybrid-bindings
```

生成器逐侧核对源身份与网格哈希，并检查全部运动帧。通过检查后，在 manifest 添加 `"hybrid_paths": {"left": "/path/to/new-hybrid-bindings/left.npz", "right": "/path/to/new-hybrid-bindings/right.npz"}`，重启查看器。访问 `?mode=hybrid&quality=surface&autoplay=1&loop=1`。左右仍为 V1/V3，同帧同相机；默认输出分辨率与单独 GS 一致，页面中的双视图尺寸不变。

混合画质默认“表面柔化”（`surface`）：沿用本侧 GS 的原生采样与外观，再根据透明度、归一化深度和颜色差异进行 3×3 联合双边去噪。透明度低于 0.98、明显颜色边缘及相对深度断裂超过 2% 的区域不进行平滑；邻近边界的强度连续衰减，每通道变化最多 0.035。它只处理显示图像，不改高斯属性、Mesh 绑定或运动，不使用跨帧图像混合，不能修复源几何的凹凸与缺失。

“GS 原生外观”（`native`）关闭去噪，便于与单独 GS 对照；“旧版均衡”（`balanced`）和“旧版高采样”（`detail`）保留 2 倍与 3 倍采样。高采样会缩小原光栅器在最终像素中的滤波覆盖，可能让源噪点更明显，不能仅凭接近高采样参考就判断视觉质量提升。所有模式、画质的缓存隔离。默认视角的 `surface` 和 `native` 各预缓存全部 400 帧；新视角仍需实时渲染，不能保证任意分辨率实时 60 FPS。显示去噪不改变原事件验收结论。

验收命令：`scripts/mesh/evaluate_hybrid_surface.py --comparison <manifest> --bindings <绑定目录> --output <新目录>`。三个视角各 48 个时刻直接比较同分辨率的单独 GS、旧版三倍混合和表面柔化；输出对比动画、原生绑定一致性、受保护区域变化、强边缘保留和局部高频变化。高频变化只是图像粗糙度代理，不能当作真实几何误差或感知质量真值。加 `--probe` 可先检查第 100 帧。旧版采样实验仍可用 `scripts/viewer/evaluate_viewer_hybrid.py --detail` 复现，其 4 倍采样参考只衡量采样接近程度。

### 4. 测试

```bash
python -m unittest discover -s tests -v
node --test tests/viewer/test_viewer_playback.cjs
```

浏览器端到端测试位于 `scripts/viewer/test_gaussian_viewer_ui.cjs`，需要额外安装 `playwright-core` 并配置 Chrome 路径。

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
python scripts/appearance/check_lego_surface_detail_release.py \
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



返回 [复现指南](05-reproduction.md)。
