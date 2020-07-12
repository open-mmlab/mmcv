<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/mmcv-logo.png" width="300"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv) [![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions) [![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv) [![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

## Introduction

MMCV is a foundational python library for computer vision research and supports many
research projects in MMLAB as below:

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

### Install with pip

a. Install the lite version.

```python
pip install mmcv
```

b. Install the full version.

We provide the pre-built mmcv package with different PyTorch and CUDA versions to simplify the building.

<table class="docutils"><tbody><tr><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.5</th><th valign="bottom" align="left" width="100">torch 1.4</th><th valign="bottom" align="left" width="100">torch 1.3</th></tr>
<tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"> </td> <td align="left"> </td> </tr>
<tr><td align="left">10.1</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td></tr>
<tr><td align="left">10.0</td><td align="left"> </td><td align="left"> </td> <td align="left"> </td> </tr>
<tr><td align="left">9.2</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td></tr>
<tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==latest+torch1.5.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.4.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==latest+torch1.3.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> </tr>
</tbody></table>

Another way is to compile locally by running

```python
pip install mmcv-full
```

Note that the local compiling may take up to 10 mins.

### Install from source

After cloning the repo with

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

You can either

- install the lite version

    ```bash
    pip install -e .
    ```

- install the full version

    ```bash
    MMCV_WITH_OPS=1 pip install -e .
    ```

If you are on macOS, add the following environment variables before the installing command.

```bash
CC=lang CXX=clang++ CFLAGS='-stdlib=libc++'
```

e.g.,

```bash
CC=lang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .
```

Note: If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.
