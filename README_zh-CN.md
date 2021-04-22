<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/mmcv-logo.png" width="300"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv) [![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions) [![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv) [![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[English](README.md) | 简体中文

## 简介

MMCV 是一个面向计算机视觉的基础库，它支持了很多开源项目，例如：

- [MMDetection](https://github.com/open-mmlab/mmdetection): 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): 语义分割工具箱
- [MMEditing](https://github.com/open-mmlab/mmediting): 图像视频编辑工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): 姿态估计工具箱
- [MMAction2](https://github.com/open-mmlab/mmaction2): 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): 一体化视频目标感知平台
- [MMClassification](https://github.com/open-mmlab/mmclassification): 图像分类工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): 全流程文字检测识别理解工具包

MMCV 提供了如下众多功能：

- 通用的 IO 接口
- 图像和视频处理
- 图像和标注结果可视化
- 常用小工具（进度条，计时器等）
- 基于 PyTorch 的通用训练框架
- 多种 CNN 网络结构
- 高质量实现的常见 CUDA 算子

如想了解更多特性和使用，请参考[文档](http://mmcv.readthedocs.io/en/latest)。

提示: MMCV 需要 Python 3.6 以上版本。

## 安装

MMCV 有两个版本：

- **mmcv**: 精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。
- **mmcv-full**: 完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。

**注意**: 请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。

a. 安装精简版

```python
pip install mmcv
```

b. 安装完整版

在安装 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 PyTorch 官方[文档](https://pytorch.org/)。

我们提供了不同 PyTorch 和 CUDA 版本的 mmcv-full 预编译包，可以大大简化用户安装编译过程。强烈推荐通过预编译包来安装。

i. 安装最新版本

如下是安装最新版 ``mmcv-full`` 的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请将链接中的 ``{cu_version}`` 和 ``{torch_version}`` 根据自身需求替换成实际的版本号，例如想安装和 ``CUDA 11``、``PyTorch 1.7.0`` 兼容的最新版 ``mmcv-full``，使用如下替换过的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

如果想知道更多 CUDA 和 PyTorch 版本的命令，可以参考下面的表格，将链接中的 ``=={mmcv_version}`` 删去即可。

ii. 安装特定的版本

如下是安装特定版本 ``mmcv-full`` 的命令

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

首先请参考版本发布信息找到想要安装的版本号，将 ``{mmcv_version}`` 替换成该版本号，例如 ``1.2.2``。
然后将链接中的 ``{cu_version}`` 和 ``{torch_version}`` 根据自身需求替换成实际的版本号，例如想安装和 ``CUDA 11``、``PyTorch 1.7.0`` 兼容的 ``mmcv-full`` 1.2.2 版本，使用如下替换过的命令

```shell
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

对于更多的 PyTorch 和 CUDA 版本组合，请参考下表：

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="100">torch 1.8</th>
      <th valign="bottom" align="left" width="100">torch 1.7</th>
      <th valign="bottom" align="left" width="100">torch 1.6</th>
      <th valign="bottom" align="left" width="100">torch 1.5</th>
      <th valign="bottom" align="left" width="100">torch 1.4</th>
      <th valign="bottom" align="left" width="100">torch 1.3</th>
    </tr>
    <tr>
      <td align="left">11.1</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
       <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

除了使用预编译包之外，另一种方式是在本地进行编译，直接运行下述命令

```python
pip install mmcv-full
```

但注意本地编译可能会耗时 10 分钟以上。

c. 安装完整版并且编译 onnxruntime 的自定义算子

- 详细的指南请查看 [这里](docs/onnxruntime_op.md)。

如果想从源码编译 MMCV，请参考[该文档](https://mmcv.readthedocs.io/en/latest/build.html)。

## FAQ

如果你遇到了安装问题，CUDA 相关的问题或者 RuntimeErrors，可以首先参考[问题解决页面](https://mmcv.readthedocs.io/en/latest/trouble_shooting.html) 看是否已经有解决方案。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMCV 所作出的努力。请参考[贡献指南](CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="docs/_static/zhihu_qrcode.jpg" height="400" />  <img src="docs/_static/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
