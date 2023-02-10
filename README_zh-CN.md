<div align="center">
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/mmcv-logo.png" width="300"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![docs](https://img.shields.io/badge/docs-2.x-blue)](https://mmcv.readthedocs.io/zh_CN/2.x/)
[![platform](https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue)](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![pytorch](https://img.shields.io/badge/pytorch-1.6~1.13-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-9.2~11.7-green)](https://developer.nvidia.com/cuda-downloads)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[English](README.md) | 简体中文

## 简介

MMCV 是一个面向计算机视觉的基础库，它提供了以下功能：

- [图像和视频处理](https://mmcv.readthedocs.io/zh_CN/2.x/understand_mmcv/data_process.html)
- [图像和标注结果可视化](https://mmcv.readthedocs.io/zh_CN/2.x/understand_mmcv/visualization.html)
- [图像变换](https://mmcv.readthedocs.io/zh_CN/2.x/understand_mmcv/data_transform.html)
- [多种 CNN 网络结构](https://mmcv.readthedocs.io/zh_CN/2.x/understand_mmcv/cnn.html)
- [高质量实现的常见 CPU 和 CUDA 算子](https://mmcv.readthedocs.io/zh_CN/2.x/understand_mmcv/ops.html)

MMCV 支持多种平台，包括：

- Linux
- Windows
- macOS

如想了解更多特性和使用，请参考[文档](http://mmcv.readthedocs.io/zh_CN/2.x)。

提示：MMCV 需要 Python 3.7 以上版本。

## 安装

MMCV 有两个版本：

- **mmcv**：完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。
- **mmcv-lite**：精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。

**注意**：请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果 CUDA 可用，强烈推荐安装 mmcv`。

### 安装 mmcv

在安装 mmcv 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。如果你使用的是搭载 apple silicon 的 mac 设备，请安装 PyTorch 1.13+ 的版本。

安装 mmcv 的命令如下：

```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc1"
```

如果需要指定 mmcv 的版本，可以使用以下命令：

```bash
mim install mmcv==2.0.0rc3
```

如果发现上述的安装命令没有使用预编译包（以 `.whl` 结尾）而是使用源码包（以 `.tar.gz` 结尾）安装，则有可能是我们没有提供和当前环境的 PyTorch 版本、CUDA 版本相匹配的 mmcv 预编译包，此时，你可以[源码安装 mmcv](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/build.html)。

<details>
<summary>使用预编译包的安装日志</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv<br />
<b>Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0rc3-cp38-cp38-manylinux1_x86_64.whl</b>

</details>

<details>
<summary>使用源码包的安装日志</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv==2.0.0rc3<br />
<b>Downloading mmcv-2.0.0rc3.tar.gz</b>

</details>

更多安装方式请参考[安装文档](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。

### 安装 mmcv-lite

如果你需要使用和 PyTorch 相关的模块，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。

```bash
pip install -U openmim
mim install "mmcv-lite>=2.0.0rc1"
```

## FAQ

如果你遇到了安装问题或者运行时问题，请查看[问题解决页面](https://mmcv.readthedocs.io/zh_CN/2.x/faq.html)是否已有解决方案。如果问题仍然没有解决，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

## 引用

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 MMCV：

```latex
@misc{mmcv,
    title={{MMCV: OpenMMLab} Computer Vision Foundation},
    author={MMCV Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcv}},
    year={2018}
}
```

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMCV 所作出的努力。请参考[贡献指南](CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 许可证

MMCV 目前以 Apache 2.0 的许可证发布，但是其中有一部分功能使用其他的许可证。我们在[许可证](LICENSES.md)中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的[知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的[官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=K0QI8ByU)，或添加微信小助手“OpenMMLabwx”加入官方交流微信群。

<div align="center">
<img src="https://user-images.githubusercontent.com/25839884/205870927-39f4946d-8751-4219-a4c0-740117558fd7.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/25839884/203904835-62392033-02d4-4c73-a68c-c9e4c1e2b07f.jpg" height="400" /> <img src="https://user-images.githubusercontent.com/25839884/205872898-e2e6009d-c6bb-4d27-8d07-117e697a3da8.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
