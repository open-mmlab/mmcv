<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/mmcv-logo.png" width="300"/>
</div>


[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv) [![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions) [![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv) [![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)


## Introduction

MMCV is a foundational python library for computer vision research and supports many
research projects in MMLAB, such as [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMAction](https://github.com/open-mmlab/mmaction).

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
* **mmcv**: lite, without CUDA ops, similar to mmcv<=1.0.0.
* **mmcv-full**: comprehensive, with various CUDA ops out of box.

a. Install lite version **mmcv** without CUDA ops.

Install from PyPI,

```bash
pip install mmcv
```

or install from source

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
```

If you are on macOS, replace the last command with

```bash
CC=lang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

b. Install full version **mmcv-full** with CUDA ops.

Install from PyPI,

```bash
pip install mmcv-full
```

or install from source

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
```

Note that the building process may take up to 10 mins. We also provide pre-built **mmcv-full** with corresponding Pytorch and CUDA versions to simplify the building.

<table class="docutils"><tbody><tr><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.5</th><th valign="bottom" align="left" width="100">torch 1.4</th><th valign="bottom" align="left" width="100">torch 1.3</th></tr>
<tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"> </td> <td align="left"> </td> </tr>
<tr><td align="left">10.1</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==1.0rc1+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.3.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td></tr>
<tr><td align="left">10.0</td><td align="left"> </td><td align="left"> </td> <td align="left"> </td> </tr>
<tr><td align="left">9.2</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==1.0rc1+torch1.5.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.4.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.3.0+cu92 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td></tr>
<tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full==1.0rc1+torch1.5.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.4.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full==1.0rc1+torch1.3.0+cpu -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
</code></pre> </details> </td> </tr>
</tbody></table>

Note: If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.
