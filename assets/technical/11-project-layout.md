# 项目目录与入口索引

根目录保留主要运行入口、说明、许可和基础依赖。辅助命令进入按功能分类的目录，公共 Python 模块使用完整包路径导入。这次整理只调整本地文件组织与对应引用，不修改训练算法、模型权重或历史实验快照。

## 根目录保留

| 文件／目录 | 用途 |
| --- | --- |
| `readme.md`、`LICENSE`、`requirements.txt` | 项目概览、许可和基础依赖 |
| `train.py`、`train_gui.py`、`edit_gui.py` | 原有训练及交互编辑入口 |
| `render.py`、`viewer.py` | 原有渲染和浏览器服务入口 |
| `scene/`、`arguments/`、`utils/`、`gaussian_renderer/` | 核心模型、参数、公共工具及渲染模块 |
| `scripts/`、`tests/` | 功能脚本和对应测试 |
| `assets/`、`viewer/` | 文档素材和前端 |
| `data/`、`outputs/` | 本地数据和实验结果，保留已有路径 |
| `requirements/` | 各功能的可选依赖 |
| `submodules/`、`lpipsPyTorch/`、`data_tools/` | 原有外部依赖及数据工具，保留现有结构 |

## 新旧路径

| 原根目录路径 | 当前路径 |
| --- | --- |
| `cam_utils.py` | `utils/cam_utils.py` |
| `lap_deform.py` | `utils/lap_deform.py` |
| `train_gui_utils.py` | `utils/train_gui_utils.py` |
| `convert.py` | `scripts/preprocessing/convert.py` |
| `full_eval.py` | `scripts/evaluation/full_eval.py` |
| `metrics.py` | `scripts/evaluation/metrics.py` |
| `export_gaussian_sequence.py` | `scripts/export/export_gaussian_sequence.py` |
| `export_lego_sequence.py` | `scripts/export/export_lego_sequence.py` |
| `export_rigid_model.py` | `scripts/export/export_rigid_model.py` |
| `pack_gaussian_sequence.py` | `scripts/export/pack_gaussian_sequence.py` |
| `render_ply.py` | `scripts/rendering/render_ply.py` |
| `render_rotation.py` | `scripts/rendering/render_rotation.py` |
| `render_gaussian_sequence.py` | `scripts/rendering/render_gaussian_sequence.py` |
| `requirements-autodl.txt` | `requirements/autodl.txt` |
| `requirements-event.txt` | `requirements/event.txt` |
| `requirements-mesh.txt` | `requirements/mesh.txt` |
| `requirements-viewer.txt` | `requirements/viewer.txt` |
| `train_gui.sh` | `scripts/training/train_gui.sh` |
| `fan_dynamic_gaussians.ply` | `data/gaussians/fan_dynamic_gaussians.ply` |

## 命令行使用

以下命令从项目根目录运行，数据路径按实际情况填写：

```bash
python scripts/export/export_rigid_model.py --help
python scripts/export/export_gaussian_sequence.py --help
python scripts/export/pack_gaussian_sequence.py --help
python scripts/rendering/render_ply.py --help
python scripts/rendering/render_gaussian_sequence.py --help
python scripts/preprocessing/convert.py --help
python scripts/evaluation/full_eval.py --help
pip install -r requirements/event.txt
pip install -r requirements/viewer.txt
```

移动后的 Python 命令会根据自身位置将项目根目录加入模块搜索路径，支持直接执行或 `python -m scripts.export.pack_gaussian_sequence`。普通相对输入／输出路径仍相对于调用时的工作目录；从其他目录调用时请使用明确的数据路径。

GUI 启动示例移至 `scripts/training/train_gui.sh`，会先切回项目根目录。脚本中的 `YOUR/PATH/TO/DATASET` 是需要填写的示例路径，不能直接当作已配置训练任务。评测脚本使用当前 Python 解释器及项目内的绝对子脚本路径。

Python 引用改为 `utils.cam_utils`、`utils.lap_deform`、`utils.train_gui_utils`、`scripts.rendering.render_ply` 和 `scripts.export.*`。若其他未纳入仓库的笔记本或个人脚本使用旧顶层导入，需要按本表更新；根目录不保留同名兼容副本。

## 数据与历史结果

`fan_dynamic_gaussians.ply` 仅移动到 `data/gaussians/`，内容哈希保持不变。`data/`、`outputs/` 内已有数据及实验归档保持原样，不改写旧源码快照和审计哈希。归档中的历史运行命令仍代表当时布局，复跑时按本页迁移入口。

本轮根目录的 Python 缓存与 Finder 元数据移入本地整理记录，不删除。Python／Finder 运行后可能重新生成这些隐藏或缓存文件，这是正常行为，不应移动 `.git` 等版本控制目录。

## 整理记录与撤销

本次移动清单、修改前备份与核验结果在 `outputs/local-organization/20261026-100327/`。`manifest.json` 保存新旧路径，`before/` 保存修改前源码与文档；PLY 没有额外复制，仍保存在新的数据路径。撤销脚本会先核对文件是否在整理后被修改，遇到冲突停止而不覆盖新工作。

只检查是否可以撤销（不修改文件）：

```bash
python outputs/local-organization/20261026-100327/undo.py
```

需要恢复原布局时，在上述命令末尾加 `--apply`。撤销会保留整理后版本的副本，已重新生成的缓存不会被覆盖；此脚本不会更改 Git 提交记录。

本轮验证：375 个 Python 文件通过语法解析，相关 Shell 脚本通过语法检查；41 项定向回归测试中 40 项通过、1 项 CUDA 测试跳过。由于本机默认 Python 环境存在依赖问题，回归使用独立临时环境执行，未修改原环境；未执行 GPU 训练或完整渲染。

目录调整不自动提交或推送 GitHub，也不更改服务器代码。
