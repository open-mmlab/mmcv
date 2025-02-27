<div align="center">
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png" width="300"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![platform](https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue)](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![pytorch](https://img.shields.io/badge/pytorch-1.8~2.0-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-10.1~11.8-green)](https://developer.nvidia.com/cuda-downloads)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[📘Documentation](https://mmcv.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmcv/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Highlights

The OpenMMLab team released a new generation of training engine [MMEngine](https://github.com/open-mmlab/mmengine) at the World Artificial Intelligence Conference on September 1, 2022. It is a foundational library for training deep learning models. Compared with MMCV, it provides a universal and powerful runner, an open architecture with a more unified interface, and a more customizable training process.

MMCV v2.0.0 official version was released on April 6, 2023. In version 2.x, it removed components related to the training process and added a data transformation module. Also, starting from 2.x, it renamed the package names **mmcv** to **mmcv-lite** and **mmcv-full** to **mmcv**. For details, see [Compatibility Documentation](docs/en/compatibility.md).

MMCV will maintain both [1.x](https://github.com/open-mmlab/mmcv/tree/1.x) (corresponding to the original [master](https://github.com/open-mmlab/mmcv/tree/master) branch) and **2.x** (corresponding to the **main** branch, now the default branch) versions simultaneously. For details, see [Branch Maintenance Plan](README.md#branch-maintenance-plan).

## Introduction

MMCV is a foundational library for computer vision research and it provides the following functionalities:

- [Image/Video processing](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_process.html)
- [Image and annotation visualization](https://mmcv.readthedocs.io/en/latest/understand_mmcv/visualization.html)
- [Image transformation](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_transform.html)
- [Various CNN architectures](https://mmcv.readthedocs.io/en/latest/understand_mmcv/cnn.html)
- [High-quality implementation of common CPU and CUDA ops](https://mmcv.readthedocs.io/en/latest/understand_mmcv/ops.html)

It supports the following systems:

- Linux
- Windows
- macOS

See the [documentation](http://mmcv.readthedocs.io/en/latest) for more features and usage.

Note: MMCV requires Python 3.7+.

## Installation

There are two versions of MMCV:

- **mmcv**: comprehensive, with full features and various CUDA/hip(ROCm) ops out of the box. It takes longer time to build.
- **mmcv-lite**: lite, without CUDA/hip(ROCm) ops but all other features, similar to mmcv\<1.0.0. It is useful when you do not need those CUDA/hip(ROCm) ops.

**Note**: Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other. `Installing the full version is highly recommended if CUDA/hip(ROCm) is available`.

### Install mmcv

Before installing mmcv, make sure that PyTorch has been successfully installed following the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation). For apple silicon users, please use PyTorch 1.13+.

The command to install mmcv:

```bash
pip install -U openmim
mim install mmcv
```

If you need to specify the version of mmcv, you can use the following command:

```bash
mim install mmcv==2.0.0
```

If you find that the above installation command does not use a pre-built package ending with `.whl` but a source package ending with `.tar.gz`, you may not have a pre-build package corresponding to the PyTorch or CUDA/hip(ROCm) or mmcv version, in which case you can [build mmcv from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html).

<details>
<summary>Installation log using pre-built packages</summary>
  
###### 1. For CUDA pre-built packages
Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv<br />
<b>Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0-cp38-cp38-manylinux1_x86_64.whl</b>
  

###### 2. For ROCm pre-built packages:  
The external [`mmcv-rocm-build` repository](https://github.com/Looong01/mmcv-rocm-build) provides wheels and detailed instructions on how to install MMCV for ROCm.
If you have any questions about it, please open an issue [here](https://github.com/Looong01/mmcv-rocm-build/issues).

</details>

<details>
<summary>Installation log using source packages</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv==2.0.0<br />
<b>Downloading mmcv-2.0.0.tar.gz</b>

</details>

For more installation methods, please refer to the [Installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

### Install mmcv-lite

If you need to use PyTorch-related modules, make sure PyTorch has been successfully installed in your environment by referring to the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation).

```bash
pip install -U openmim
mim install mmcv-lite
```

## FAQ

If you face some installation issues, CUDA related issues or RuntimeErrors,
you may first refer to this [Frequently Asked Questions](https://mmcv.readthedocs.io/en/latest/faq.html).

If you face installation problems or runtime issues, you may first refer to this [Frequently Asked Questions](https://mmcv.readthedocs.io/en/latest/faq.html) to see if there is a solution. If the problem is still not solved, feel free to open an [issue](https://github.com/open-mmlab/mmcv/issues).

If you face some installation issues with ROCm build, you may first refer to [this](https://github.com/Looong01/mmcv-rocm-build/issues) to see if there is a solution or feel free to open an [issue](https://github.com/Looong01/mmcv-rocm-build/issues) if not.

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

## Branch Maintenance Plan

MMCV currently has four branches, namely main, 1.x, master, and 2.x, where 2.x is an alias for the main branch, and master is an alias for the 1.x branch. The 2.x and master branches will be deleted in the future. MMCV's branches go through the following three stages:

| Phase                | Time                  | Branch                                                                                                                              | description                                                                                                                                            |
| -------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RC Period            | 2022.9.1 - 2023.4.5   | Release candidate code (2.x version) will be released on 2.x branch. Default master branch is still 1.x version                     | Master and 2.x branches iterate normally                                                                                                               |
| Compatibility Period | 2023.4.6 - 2023.12.31 | **The 2.x branch has been renamed to the main branch and set as the default branch**, and 1.x branch will correspond to 1.x version | We still maintain the old version 1.x, respond to user needs, but try not to introduce changes that break compatibility; main branch iterates normally |
| Maintenance Period   | From 2024/1/1         | Default main branch corresponds to 2.x version and 1.x branch is 1.x version                                                        | 1.x branch is in maintenance phase, no more new feature support; main branch is iterating normally                                                     |

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
