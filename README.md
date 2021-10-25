<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/mmcv-logo.png" width="300"/>
</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/) [![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv) [![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions) [![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv) [![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

English | [简体中文](README_zh-CN.md)

## Introduction

MMCV is a foundational library for computer vision research and supports many
research projects as below:

- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.

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

- **mmcv-full**: comprehensive, with full features and various CUDA ops out of box. It takes longer time to build.
- **mmcv**: lite, without CUDA ops but all other features, similar to mmcv<1.0.0. It is useful when you do not need those CUDA ops.

**Note**: Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other. `Installing the full version is highly recommended if CUDA is available`.

a. Install the full version.

Before installing mmcv-full, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/).

We provide pre-built mmcv packages (recommended) with different PyTorch and CUDA versions to simplify the building.

i. Install the latest version.

The rule for installing the latest ``mmcv-full`` is as follows:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example,
to install the latest ``mmcv-full`` with ``CUDA 11.1`` and ``PyTorch 1.9.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

Note: mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well. For example, if your PyTorch version is 1.8.1 and CUDA version is 11.1, you can use the following command to install mmcv-full.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

For more details, please refer the the following tables and delete ``=={mmcv_version}``.

ii. Install a specified version.

The rule for installing a specified ``mmcv-full`` is as follows:

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

First of all, please refer to the Releases and replace ``{mmcv_version}`` a specified one. e.g. ``1.3.9``.
Then replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired versions. For example,
to install ``mmcv-full==1.3.9`` with ``CUDA 11.1`` and ``PyTorch 1.9.0``, use the following command:

```shell
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

For more details, please refer the the following tables.

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="100">torch 1.9</th>
      <th valign="bottom" align="left" width="100">torch 1.8</th>
      <th valign="bottom" align="left" width="100">torch 1.7</th>
      <th valign="bottom" align="left" width="100">torch 1.6</th>
      <th valign="bottom" align="left" width="100">torch 1.5</th>
      <th valign="bottom" align="left" width="100">torch 1.4</th>
      <th valign="bottom" align="left" width="100">torch 1.3</th>
    </tr>
    <tr>
      <td align="left">11.1</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

Another way is to compile locally by running

```python
pip install mmcv-full
```

Note that the local compiling may take up to 10 mins.

b. Install the lite version.

```python
pip install mmcv
```

c. Install full version with custom operators for onnxruntime

- Check [here](docs/deployment/onnxruntime_op.md) for detailed instruction.

If you would like to build MMCV from source, please refer to the [guide](https://mmcv.readthedocs.io/en/latest/get_started/build.html).

## FAQ

If you face some installation issues, CUDA related issues or RuntimeErrors,
you may first refer to this [Frequently Asked Questions](https://mmcv.readthedocs.io/en/latest/faq.html).

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmcv,
    title={{MMCV: OpenMMLab} Computer Vision Foundation},
    author={MMCV Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcv}},
    year={2018}
}
```

## Contributing

We appreciate all contributions to improve MMCV. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## License

MMCV is released under the Apache 2.0 license, while some specific operations in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.
