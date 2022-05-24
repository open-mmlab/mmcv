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

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmcv.readthedocs.io/zh_CN/latest/)
[![platform](https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue)](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[English](README.md) | 简体中文

## 简介

MMCV 是一个面向计算机视觉的基础库，它提供了以下功能：

- 通用的 IO 接口
- 图像和视频处理
- 图像和标注结果可视化
- 常用小工具（进度条，计时器等）
- 基于 PyTorch 的通用训练框架
- 多种 CNN 网络结构
- 高质量实现的常见 CUDA 算子

MMCV 支持多种平台，包括：

- Linux
- Windows
- macOS

如想了解更多特性和用法，请参考[文档](http://mmcv.readthedocs.io/zh_CN/latest)。

提示: MMCV 需要 Python 3.6 以上版本。

## 安装

MMCV 有两个版本：

- **mmcv-full**: 完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。
- **mmcv**: 精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。

本文档只提供 mmcv-full 的安装教程，如需安装精简版的 mmcv，请参考[安装文档](todo)。另外，在安装 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 PyTorch [官方文档](https://pytorch.org/)。

在 Linux 和 Windows 平台安装 mmcv-full 的命令如下，如需在 macOS 平台安装 mmcv-full，请参考[源码安装 mmcv-full](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

```bash
pip install openmim
mim install mmcv-full
```

如果需要指定 mmcv-full 的版本，可以使用以下命令

```bash
mim install mmcv-full==1.5.0
```

需要注意的是，你需要参考下表查看 mmcv-full 提供对应 PyTorch 和 CUDA 版本的最低版本。

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="120">torch 1.11.x</th>
      <th valign="bottom" align="left" width="120">torch 1.10.x</th>
      <th valign="bottom" align="left" width="120">torch 1.9.x</th>
      <th valign="bottom" align="left" width="120">torch 1.8.x</th>
      <th valign="bottom" align="left" width="120">torch 1.7.x</th>
      <th valign="bottom" align="left" width="120">torch 1.6.x</th>
      <th valign="bottom" align="left" width="120">torch 1.5.x</th>
    </tr>
    <tr>
      <td align="left">11.5</td><td align="left">1.4.7</td><td align="left"></td>
      <td align="left"></td><td align="left"></td><td align="left"></td>
      <td align="left"></td><td align="left"></td>
    </tr>
    <tr>
      <td align="left">11.3</td><td align="left">1.4.7</td><td align="left">1.3.16</td>
      <td align="left"></td><td align="left"></td><td align="left"></td>
      <td align="left"></td><td align="left"></td>
    </tr>
    <tr>
      <td align="left">11.1</td><td align="left"></td><td align="left">1.3.16</td>
      <td align="left">1.3.9</td><td align="left">1.3.0</td><td align="left"></td>
      <td align="left"></td><td align="left"></td>
    </tr>
    <tr>
      <td align="left">11.0</td><td align="left"></td><td align="left"></td>
      <td align="left"></td><td align="left"></td><td align="left">1.2.0</td>
      <td align="left"></td><td align="left"></td>
    </tr>
    <tr>
      <td align="left">10.2</td><td align="left">1.4.7 (Linux only)</td><td align="left">1.3.16</td>
      <td align="left">1.3.9</td><td align="left">1.3.0</td><td align="left">1.2.0</td>
      <td align="left">1.0.5</td><td align="left">1.0.0</td>
    </tr>
    <tr>
      <td align="left">10.1</td><td align="left"></td><td align="left"></td>
      <td align="left"></td><td align="left">1.3.0</td><td align="left">1.2.0</td>
      <td align="left">1.0.5</td><td align="left">1.0.0</td>
    </tr>
    <tr>
      <td align="left">9.2</td><td align="left"></td><td align="left"></td>
      <td align="left"></td><td align="left"></td><td align="left">1.2.0 (Linux only)</td>
      <td align="left">1.0.5 (Linux only)</td><td align="left">1.0.0</td>
    </tr>
    <tr>
      <td align="left">cpu</td><td align="left">1.4.7</td><td align="left">1.3.16</td>
      <td align="left">1.3.9</td><td align="left">1.3.0</td><td align="left">1.2.0</td>
      <td align="left">1.0.5</td><td align="left">1.0.0</td>
    </tr>
  </tbody>
</table>

**注意**：mmcv-full>=1.4.0 才开始提供 Windows 平台的预编译包。

如果想从源码编译 MMCV，请参考[源码安装 mmcv-full](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

## 支持的部分开源项目

- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
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

## FAQ

如果你遇到了安装问题，CUDA 相关的问题或者 RuntimeErrors，可以首先参考[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html)查看是否已经有解决方案。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMCV 所作出的努力。请参考[贡献指南](CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 许可证

`MMCV` 目前以 Apache 2.0 的许可证发布，但是其中有一部分功能并不是使用的 Apache2.0 许可证，我们在 [许可证](LICENSES.md) 中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=3ijNTqfg)，或添加微信小助手”OpenMMLabwx“加入官方交流微信群。

<div align="center">
<img src="docs/en/_static/zhihu_qrcode.jpg" height="400" />  <img src="docs/en/_static/qq_group_qrcode.jpg" height="400" /> <img src="docs/en/_static/wechat_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
