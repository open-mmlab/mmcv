<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/mmcv-logo.png" width="300"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv) [![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions) [![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv) [![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

## Introduction

MMCV is a foundational python library for computer vision research and supports many
research projects as below:

- [MMDetection](https://github.com/open-mmlab/mmdetection): Detection toolbox and benchmark
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): General 3D object detection toolbox and benchmark
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): Semantic segmentation toolbox and benchmark
- [MMEditing](https://github.com/open-mmlab/mmediting): Image and video editing toolbox
- [MMPose](https://github.com/open-mmlab/mmpose): Pose estimation toolbox and benchmark
- [MMAction2](https://github.com/open-mmlab/mmaction2): Action understanding toolbox and benchmark
- [MMClassification](https://github.com/open-mmlab/mmclassification): Image classification toolbox and benchmark

It provides the following functionalities.

- Universal IO APIs
- Image/Video processing
- Image and annotation visualization
- Useful utilities (progress bar, timer, ...)
- PyTorch runner with hooking mechanism
- Various CNN architectures
- High-quality implementation of common CUDA ops

See the [documentation](http://mmcv.readthedocs.io/en/latest) for more features and usage.

Note: MMCV requires Python 3.6+.

## Installation

There are two versions of MMCV:

- **mmcv**: lite, without CUDA ops but all other features, similar to mmcv<1.0.0. It is useful when you do not need those CUDA ops.
- **mmcv-full**: comprehensive, with full features and various CUDA ops out of box. It takes longer time to build.

**Note**: Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other.

a. Install the lite version.

```python
pip install mmcv
```

b. Install the full version.

Before installing mmcv-full, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/).

We provide pre-built mmcv packages (recommended) with different PyTorch and CUDA versions to simplify the building.

**For pip < 20.3, please refer to the following table:**
<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="100">torch 1.7</th>
      <th valign="bottom" align="left" width="100">torch 1.6</th>
      <th valign="bottom" align="left" width="100">torch 1.5</th>
      <th valign="bottom" align="left" width="100">torch 1.4</th>
      <th valign="bottom" align="left" width="100">torch 1.3</th>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.7.0+cu110 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.7.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.6.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.5.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.7.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.7.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.6.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.7.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.6.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

**For pip >= 20.3, please refer to the following table:**

(NOTE: Please refer to the Releases and replace ``{mmcv_version}`` a specified one. e.g. ``1.2.1``; BTW, specifying version using ``latest`` is currently disabled.)
<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="100">torch 1.7</th>
      <th valign="bottom" align="left" width="100">torch 1.6</th>
      <th valign="bottom" align="left" width="100">torch 1.5</th>
      <th valign="bottom" align="left" width="100">torch 1.4</th>
      <th valign="bottom" align="left" width="100">torch 1.3</th>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

Another way is to compile locally by running

```python
pip install mmcv-full
```

Note that the local compiling may take up to 10 mins.

If you would like to build MMCV from source, please refer to the [guide](https://mmcv.readthedocs.io/en/latest/build.html).
