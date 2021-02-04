# TensorRT Plugins for custom operators in MMCV (Experimental)

<!-- TOC -->

- [TensorRT Plugins for custom operators in MMCV (Experimental)](#tensorrt-plugins-for-custom-operators-in-mmcv-experimental)
  - [Introduction](#introduction)
  - [List of TensorRT plugins supported in MMCV](#list-of-tensorrt-plugins-supported-in-mmcv)
  - [How to build TensorRT plugins in MMCV](#how-to-build-tensorrt-plugins-in-mmcv)
    - [Prerequisite](#prerequisite)
    - [Build on Linux](#build-on-linux)
  - [Create TensorRT engine and run inference in python](#create-tensorrt-engine-and-run-inference-in-python)
  - [How to add a TensorRT plugin for custom op in MMCV](#how-to-add-a-tensorrt-plugin-for-custom-op-in-mmcv)
    - [Main procedures](#main-procedures)
    - [Reminders](#reminders)
  - [Known Issues](#known-issues)
  - [References](#references)

<!-- TOC -->

## Introduction

**NVIDIA TensorRT** is a software development kit(SDK) for high-performance inference of deep learning models. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. Please check its [developer's website](https://developer.nvidia.com/tensorrt) for more information.
To ease the deployment of trained models with custom operators from `mmcv.ops` using TensorRT, a series of TensorRT plugins are included in MMCV.

## List of TensorRT plugins supported in MMCV

|   ONNX Operator   |    TensorRT Plugin    | Note  |
| :---------------: | :-------------------: | :---: |
|     RoiAlign      |     MMCVRoiAlign      |   Y   |
|     ScatterND     |       ScatterND       |   Y   |
| NonMaxSuppression | MMCVNonMaxSuppression |  WIP  |

Notes

- All plugins listed above are developed on TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0

## How to build TensorRT plugins in MMCV

### Prerequisite

- Clone repository

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

- Install TensorRT

Download the corresponding TensorRT build from [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download).

For example, for Ubuntu 16.04 on x86-64 with cuda-10.2, the downloaded file is `TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz`.

Then, install as below:

```bash
cd ~/Downloads
tar -xvzf TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
export TENSORRT_DIR=`pwd`/TensorRT-7.2.1.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_DIR/lib
```

Install python packages: tensorrt, graphsurgeon, onnx-graphsurgeon

```bash
pip install $TENSORRT_DIR/python/tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
pip install $TENSORRT_DIR/onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
pip install $TENSORRT_DIR/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
```

For more detailed infomation of installing TensorRT using tar, please refer to [Nvidia' website](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-721/install-guide/index.html#installing-tar).

### Build on Linux

```bash
cd mmcv # to MMCV root directory
MMCV_WITH_OPS=1 MMCV_WITH_TRT=1 pip install -e .
```

## Create TensorRT engine and run inference in python

Here is an example.

```python
import torch
import onnx

from mmcv.tensorrt import (TRTWraper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

onnx_file = 'sample.onnx'
trt_file = 'sample.trt'
onnx_model = onnx.load(onnx_file)

# Model input
inputs = torch.rand(1, 3, 224, 224).cuda()
# Model input shape info
opt_shape_dict = {
    'input': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)]
}

# Create TensorRT engine
max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    max_workspace_size=max_workspace_size)

# Save TensorRT engine
save_trt_engine(trt_engine, trt_file)

# Run inference with TensorRT
trt_model = TRTWraper(trt_file, ['input'], ['output'])

with torch.no_grad():
    trt_outputs = trt_model({'input': inputs})
    output = trt_outputs['output']

```

## How to add a TensorRT plugin for custom op in MMCV

### Main procedures

Below are the main steps:

1. Add c++ header file
2. Add c++ source file
3. Add cuda kernel file
4. Register plugin in `trt_plugin.cpp`
5. Add unit test in `tests/test_ops/test_tensorrt.py`

**Take RoIAlign plugin `roi_align` for example.**

1. Add header `trt_roi_align.hpp` to TensorRT include directory `mmcv/ops/csrc/tensorrt/`
2. Add source `trt_roi_align.cpp` to TensorRT source directory `mmcv/ops/csrc/tensorrt/plugins/`
3. Add cuda kernel `trt_roi_align_kernel.cu` to TensorRT source directory `mmcv/ops/csrc/tensorrt/plugins/`
4. Register `roi_align` plugin in [trt_plugin.cpp](https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/tensorrt/plugins/trt_plugin.cpp)

    ```c++
    #include "trt_plugin.hpp"

    #include "trt_roi_align.hpp"

    REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);

    extern "C" {
    bool initLibMMCVInferPlugins() { return true; }
    }  // extern "C"
    ```

5. Add unit test into `tests/test_ops/test_tensorrt.py`
   Check [here](https://github.com/open-mmlab/mmcv/blob/master/tests/test_ops/test_tensorrt.py) for examples.

### Reminders

- Some of the [custom ops](https://mmcv.readthedocs.io/en/latest/ops.html) in `mmcv` have their cuda implementations, which could be refered.

## Known Issues

- None

## References

- [Developer guide of Nvidia TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)
- [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
- [TensorRT python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
- [TensorRT c++ plugin API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin.html)
