[English](README.md) · **简体中文**

<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="dino.py 标志">
</p>

<h1 align="center">dino.py</h1>

<p align="center"><strong>在 ggml 上直接运行 DINOv2 / DINOv3 视觉特征，推理阶段不需要 PyTorch。</strong></p>

<p align="center">把 GGUF 权重交给它，DINO ViT 的整个前向计算都由一张手写的 ggml 计算图完成：输入 NumPy 图像，输出全局 CLS 描述子和逐 patch 特征图。</p>

这是一个实验性的参考实现，不是打包好的应用。所有脚本都不长，推理链路只依赖 `ggml` 的 Python 绑定、NumPy、Pillow 和 scikit-learn。

**依赖**

| 仓库部分 | 需要 |
| --- | --- |
| 推理（`gg.py`、`dino2.py`、`dino3.py`、`cutvid.py`） | 提供 `ggml` 和 `ggml.utils` 模块的 `ggml` 绑定、`numpy`、`pillow`、`scikit-learn` |
| `cut.py` | `PATH` 上的 `dinov2` 可执行文件 |
| 转换工具（`convert_*.py`、`llama_quantize.py`） | `torch`、`transformers`、`gguf` |

## 快速开始

### 1. 安装

需要 Python 3.12、`ggml` 的 Python FFI 绑定，以及一个 GGUF 模型。

```bash
git clone https://github.com/ZisIsNotZis/dino.py
cd dino.py
python3.12 -m venv .venv
. .venv/bin/activate
pip install numpy pillow scikit-learn
```

`ggml` 绑定请按你的平台单独安装，本仓库不附带，也没有一个能假定的 PyPI 包名；`gg.py` 只要求编译好的绑定里 `ggml` 和 `ggml.utils` 两个模块可导入。仓库也不包含模型权重和依赖锁文件。

### 2. 指向模型

`dino2.py` 和 `dino3.py` 的 `run()` 里写了本地示例路径。改掉 `model=` 的默认值，或者直接调用时传入路径：

```python
cls, grid = run(image[None], model="/path/to/dinov2-small-f16.gguf")
```

需要 CUDA 后端就传 `gpu=True`。CPU 路径默认用 28 个线程，机器不同的话在 `gg.init` 里改。

### 3. 运行

```bash
python dino2.py photo.jpg   # DINOv2
python dino3.py photo.jpg   # DINOv3
```

每张输入图旁边会生成原始 float32 特征文件，命名形如 `<图片><dino2|dino3>.<各维大小>f`，后缀就是逗号连接的数组形状；`cut.py` 正是靠这个后缀拼出要读回的 float32 特征文件名，`cutvid.py` 则直接调用 `run()`。两个脚本都支持一次传入多张图。设置 `VIS=1` 还会额外写一张 PCA 可视化 JPEG：

```bash
VIS=1 python dino2.py photo.jpg
```

### 在 Python 里调用

```python
from PIL import Image
import numpy as np
from dino2 import run

image = np.asarray(Image.open("photo.jpg").convert("RGB"))
cls, grid = run(image[None], model="/path/to/dinov2-small-f16.gguf")

print(cls.shape)    # 全局描述子，宽度取决于 checkpoint
print(grid.shape)   # 逐 patch 特征，用于稠密任务
```

## 现在能做什么

- **DINOv2 ViT 推理**：register token、融合或分离的 QKV 权重、LayerScale、GELU MLP 和最终 LayerNorm，全部显式搭成 ggml 计算图。
- **DINOv3 ViT 推理**：2D RoPE、SwiGLU、register token 和 DINOv3 的归一化布局。
- **CPU 或 CUDA**：每次运行可选 ggml 后端，推理链路不经过 PyTorch。
- **全局与稠密特征**：一次前向同时拿到 CLS 描述子和 patch 特征图，裁剪与视频工具直接使用。
- **PCA 可视化**：`VIS=1` 把 patch 特征转成 RGB 显著图，方便快速检查。
- **权重转换工具**：DINOv3 官方权重转 Hugging Face 格式、HF ViT 转 GGUF、GGUF 重新量化为 f16。
- **两个应用原型**：`cut.py` 做基于 PCA 显著图的智能裁剪，`cutvid.py` 借 FFmpeg/QSV 从视频里挑关键帧。

两个原型都依赖各自的本地环境：`cut.py` 会调用 `PATH` 上的 `dinov2` 可执行文件，`cutvid.py` 需要带 Intel QSV 的 FFmpeg。

## 架构

```text
dino.py/
├── gg.py                          # 基于 ggml FFI 的 Tensor/Graph DSL
├── dino2.py                       # DINOv2 ViT 计算图 + 图像 CLI
├── dino3.py                       # DINOv3 ViT 计算图 + 图像 CLI
├── cut.py                         # 智能裁剪原型
├── cutvid.py                      # 视频关键帧原型
├── convert_dinov3_vit_to_hf.py    # DINOv3 官方 ViT -> HF
├── convert_dinov3_convnext_to_hf.py  # DINOv3 官方 ConvNeXt -> HF
├── convert_dinov3_vit_hf_to_gguf.py  # HF DINOv2 ViT -> GGUF
├── convert_hf_to_gguf.py          # 通用 HF 模型 -> GGUF
├── llama_quantize.py              # 把现有 GGUF 重新量化为 f16
├── CONTRIBUTING.md                # 贡献范围、验证要求与 PR 规则
└── assets/logo.svg
```

数据流刻意保持直接：

```text
GGUF 权重 -> gg.py 加载 -> 显式 ggml 计算图 -> ggml 后端 -> NumPy 数组
```

`gg.py` 封装了张量、视图、矩阵乘、归一化、注意力、2D RoPE 和图分配；`dino2.py` 和 `dino3.py` 只保留模型接线，适合当作把别的 ViT 移植到 ggml 的参考。

转换脚本与推理分开，需要 PyTorch、Transformers 和 `gguf`。`convert_dinov3_vit_hf_to_gguf.py` 还依赖外部的 `dinov2_inference` 模块来映射 GGML 数据类型。

## 如何开发

- 没有构建步骤、测试集或 CI。改完就在真实图片上跑一遍，检查输出形状和 `VIS=1` 的图。
- 代码需要 Python 3.12：用了 PEP 695 泛型和 PEP 701 的 f-string 语法，旧解释器无法解析。
- `NOIMP=1` 会把原地 ggml 操作换成非原地版本，适合不支持原地操作的后端。
- `gg.py` 以 `INFO` 级别打印计算图构建过程；调试时改 `basicConfig` 的级别。
- 模型路径只是示例默认值，不是配置系统。改源码或调用时显式传 `model=`。

## 如何贡献

欢迎小而集中的补丁；范围、验证要求和 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。提 issue 时请附上：

- 使用的 checkpoint 或 GGUF 类型；
- 后端（CPU/CUDA）与 Python 版本；
- 执行的命令，以及输出或报错。

目前还没有测试框架，请给出可复现的命令和期望形状。推理链路尽量不引入新依赖；两个上游 Hugging Face 转换脚本里的 Apache-2.0 头请保留。

## 路线图

**已实现**

- [x] 从 GGUF 在 ggml 上做 DINOv2 ViT 推理
- [x] 带 2D RoPE 和 SwiGLU 的 DINOv3 ViT 推理
- [x] CPU 与 CUDA 后端选择
- [x] 一次前向同时产出 CLS 描述子和 patch 特征图
- [x] PCA 特征可视化
- [x] 智能裁剪与视频关键帧原型
- [x] GGUF 转换与 f16 重新量化工具

**计划中**

- [ ] 真正的 CLI/配置路径，不用再改源码里的模型默认值
- [ ] 包结构和固定、写清楚的运行环境
- [ ] 与 PyTorch 参考实现对齐的特征一致性测试
- [ ] 理顺 DINOv3 转 GGUF 的依赖
- [ ] 支持更多 DINOv3 骨干和更广的 GGUF 覆盖

长期目标是一个小而通用的视觉推理器：给一个 GGUF 和一张图，就能得到可用的全局与稠密特征，不必背一个框架运行时。裁剪、关键帧选择、检索都应该是同一张图之上的薄层。

## 许可证

仓库自身代码使用 [AGPL-3.0-only](LICENSE)，版权归 ZisIsNotZis 所有。两个上游 Hugging Face 转换脚本 `convert_dinov3_convnext_to_hf.py` 和 `convert_dinov3_vit_to_hf.py` 保留 Apache-2.0 版权与许可头；仓库许可证不替代这些声明。
