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
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[English](README.md) | 简体中文

## 简介

MMCV 是一个面向计算机视觉的基础库，它支持了很多开源项目，例如：

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

MMCV 提供了如下众多功能：

- 通用的 IO 接口
- 图像和视频处理
- 图像和标注结果可视化
- 常用小工具（进度条，计时器等）
- 基于 PyTorch 的通用训练框架
- 多种 CNN 网络结构
- 高质量实现的常见 CUDA 算子

MMCV 支持以下的系统：

- Linux
- Windows
- macOS

如想了解更多特性和使用，请参考[文档](http://mmcv.readthedocs.io/zh_CN/latest)。

提示: MMCV 需要 Python 3.6 以上版本。

## 安装

MMCV 有两个版本：

- **mmcv-full**: 完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。
- **mmcv**: 精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。

**注意**: 请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果CUDA可用，强烈推荐安装mmcv-full`。

a. 安装完整版

在安装 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 PyTorch [官方文档](https://pytorch.org/)。

我们提供了 **Linux 和 Windows 平台** PyTorch 和 CUDA 版本组合的 mmcv-full 预编译包，可以大大简化用户安装编译过程。强烈推荐通过预编译包来安装。另外，安装完成后可以运行 [check_installation.py](.dev_scripts/check_installation.py) 脚本检查 mmcv-full 是否安装成功。

i. 安装最新版本

如下是安装最新版 `mmcv-full` 的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请将链接中的 `{cu_version}` 和 `{torch_version}` 根据自身需求替换成实际的版本号，例如想安装和 `CUDA 11.1`、`PyTorch 1.9.0` 兼容的最新版 `mmcv-full`，使用如下替换过的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**注意**: PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果你的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv-full。例如，如果你的 PyTorch 版本是 1.8.1、CUDA 版本是 11.1，你可以使用以下命令安装 mmcv-full。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

如果想知道更多 CUDA 和 PyTorch 版本的命令，可以参考下面的表格，将链接中的 `=={mmcv_version}` 删去即可。

ii. 安装特定的版本

如下是安装特定版本 `mmcv-full` 的命令

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

首先请参考版本发布信息找到想要安装的版本号，将 `{mmcv_version}` 替换成该版本号，例如 `1.3.9`。
然后将链接中的 `{cu_version}` 和 `{torch_version}` 根据自身需求替换成实际的版本号，例如想安装和 `CUDA 11.1`、`PyTorch 1.9.0` 兼容的 `mmcv-full` 1.3.9 版本，使用如下替换过的命令

```shell
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

对于更多的 PyTorch 和 CUDA 版本组合，请参考下表：

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="120">torch 1.11</th>
      <th valign="bottom" align="left" width="120">torch 1.10</th>
      <th valign="bottom" align="left" width="120">torch 1.9</th>
      <th valign="bottom" align="left" width="120">torch 1.8</th>
      <th valign="bottom" align="left" width="120">torch 1.7</th>
      <th valign="bottom" align="left" width="120">torch 1.6</th>
      <th valign="bottom" align="left" width="120">torch 1.5</th>
    </tr>
    <tr>
      <td align="left">11.5</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"></td>
      <td align="left"></code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.3</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"></td>
      <td align="left"></code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.1</td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html</code></pre> </details></td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html</code></pre> </details></td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html</code></pre> </details> </td>
       <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

**注意**：以上提供的预编译包并不囊括所有的 mmcv-full 版本，你可以点击对应链接查看支持的版本。例如，点击 [cu102-torch1.8.0](https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html)，可以看到 `cu102-torch1.8.0` 只提供了 1.3.0 及以上的 mmcv-full 版本。另外，从 `mmcv v1.3.17` 开始，我们不再提供`PyTorch 1.3 & 1.4` 对应的 mmcv-full 预编译包。你可以在 [这](./docs/zh_cn/get_started/previous_versions.md) 找到 `PyTorch 1.3 & 1.4` 对应的预编包。虽然我们不再提供 `PyTorch 1.3 & 1.4` 对应的预编译包，但是我们依然在 CI 中保证对它们的兼容持续到下一年。

**注意**：mmcv-full 没有提供 Windows 平台 `cu102-torch1.8.0` 和 `cu92-torch*` 的预编译包。

除了使用预编译包之外，另一种方式是在本地进行编译，直接运行下述命令

```python
pip install mmcv-full
```

但注意本地编译可能会耗时 10 分钟以上。

b. 安装精简版

```python
pip install mmcv
```

c. 安装完整版并且编译 onnxruntime 的自定义算子

- 详细的指南请查看[这里](docs/zh_cn/deployment/onnxruntime_op.md)。

如果想从源码编译 MMCV，请参考[该文档](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

## FAQ

如果你遇到了安装问题，CUDA 相关的问题或者 RuntimeErrors，可以首先参考[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html) 看是否已经有解决方案。

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
