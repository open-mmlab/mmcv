## MMCV中的TensorRT自定义算子 (实验性)

<!-- TOC -->

- [MMCV中的TensorRT自定义算子 (实验性)](#mmcv中的tensorrt自定义算子-实验性)
  - [介绍](#介绍)
  - [MMCV中的TensorRT插件列表](#mmcv中的tensorrt插件列表)
  - [如何编译MMCV中的TensorRT插件](#如何编译mmcv中的tensorrt插件)
    - [准备](#准备)
    - [在Linux上编译](#在linux上编译)
  - [创建TensorRT推理引擎并在python下进行推理](#创建tensorrt推理引擎并在python下进行推理)
  - [如何在MMCV中添加新的TensorRT自定义算子](#如何在mmcv中添加新的tensorrt自定义算子)
    - [主要流程](#主要流程)
    - [注意](#注意)
  - [已知问题](#已知问题)
  - [引用](#引用)

<!-- TOC -->

### 介绍

**NVIDIA TensorRT**是一个为深度学习模型高性能推理准备的软件开发工具(SDK)。它包括深度学习推理优化器和运行时，可为深度学习推理应用提供低延迟和高吞吐量。请访问[developer's website](https://developer.nvidia.com/tensorrt)了解更多信息。
为了简化TensorRT部署带有MMCV自定义算子的模型的流程，MMCV中添加了一系列TensorRT插件。

### MMCV中的TensorRT插件列表

|         ONNX算子          |                                  TensorRT插件                                   | MMCV版本 |
| :-----------------------: | :-----------------------------------------------------------------------------: | :------: |
|       MMCVRoiAlign        |              [MMCVRoiAlign](./tensorrt_custom_ops.md#mmcvroialign)              |  1.2.6   |
|         ScatterND         |                 [ScatterND](./tensorrt_custom_ops.md#scatternd)                 |  1.2.6   |
|     NonMaxSuppression     |         [NonMaxSuppression](./tensorrt_custom_ops.md#nonmaxsuppression)         |  1.3.0   |
|     MMCVDeformConv2d      |          [MMCVDeformConv2d](./tensorrt_custom_ops.md#mmcvdeformconv2d)          |  1.3.0   |
|       grid_sampler        |              [grid_sampler](./tensorrt_custom_ops.md#grid-sampler)              |  1.3.1   |
|          cummax           |                    [cummax](./tensorrt_custom_ops.md#cummax)                    |  1.3.5   |
|          cummin           |                    [cummin](./tensorrt_custom_ops.md#cummin)                    |  1.3.5   |
| MMCVInstanceNormalization | [MMCVInstanceNormalization](./tensorrt_custom_ops.md#mmcvinstancenormalization) |  1.3.5   |
| MMCVModulatedDeformConv2d | [MMCVModulatedDeformConv2d](./tensorrt_custom_ops.md#mmcvmodulateddeformconv2d) |  master  |

注意

- 以上所有算子均在 TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0 环境下开发。

### 如何编译MMCV中的TensorRT插件

#### 准备

- 克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

- 安装TensorRT

从 [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download) 下载合适的TensorRT版本。

比如，对安装了cuda-10.2的x86-64的Ubuntu 16.04，下载文件为`TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz`.

然后使用下面方式安装并配置环境

```bash
cd ~/Downloads
tar -xvzf TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
export TENSORRT_DIR=`pwd`/TensorRT-7.2.1.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_DIR/lib
```

安装python依赖: tensorrt, graphsurgeon, onnx-graphsurgeon

```bash
pip install $TENSORRT_DIR/python/tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
pip install $TENSORRT_DIR/onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
pip install $TENSORRT_DIR/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
```

想了解更多通过tar包安装TensorRT，请访问[Nvidia' website](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-721/install-guide/index.html#installing-tar).

- 安装 cuDNN

参考[Nvidia' website](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)安装 cuDNN 8。

#### 在Linux上编译

```bash
cd mmcv ## to MMCV root directory
MMCV_WITH_OPS=1 MMCV_WITH_TRT=1 pip install -e .
```

### 创建TensorRT推理引擎并在python下进行推理

范例如下：

```python
import torch
import onnx

from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

onnx_file = 'sample.onnx'
trt_file = 'sample.trt'
onnx_model = onnx.load(onnx_file)

## Model input
inputs = torch.rand(1, 3, 224, 224).cuda()
## Model input shape info
opt_shape_dict = {
    'input': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)]
}

## Create TensorRT engine
max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    max_workspace_size=max_workspace_size)

## Save TensorRT engine
save_trt_engine(trt_engine, trt_file)

## Run inference with TensorRT
trt_model = TRTWrapper(trt_file, ['input'], ['output'])

with torch.no_grad():
    trt_outputs = trt_model({'input': inputs})
    output = trt_outputs['output']

```

### 如何在MMCV中添加新的TensorRT自定义算子

#### 主要流程

下面是主要的步骤：

1. 添加c++头文件
2. 添加c++源文件
3. 添加cuda kernel文件
4. 在`trt_plugin.cpp`中注册插件
5. 在`tests/test_ops/test_tensorrt.py`中添加单元测试

**以RoIAlign算子插件`roi_align`举例。**

1. 在TensorRT包含目录`mmcv/ops/csrc/tensorrt/`中添加头文件`trt_roi_align.hpp`
2. 在TensorRT源码目录`mmcv/ops/csrc/tensorrt/plugins/`中添加头文件`trt_roi_align.cpp`
3. 在TensorRT源码目录`mmcv/ops/csrc/tensorrt/plugins/`中添加cuda kernel文件`trt_roi_align_kernel.cu`
4. 在[trt_plugin.cpp](https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/tensorrt/plugins/trt_plugin.cpp)中注册`roi_align`插件

    ```c++
    #include "trt_plugin.hpp"

    #include "trt_roi_align.hpp"

    REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);

    extern "C" {
    bool initLibMMCVInferPlugins() { return true; }
    }  // extern "C"
    ```

5. 在`tests/test_ops/test_tensorrt.py`中添加单元测试

#### 注意

- 部分MMCV中的自定义算子存在对应的cuda实现，在进行TensorRT插件开发的时候可以参考。

### 已知问题

- 无

### 引用

- [Developer guide of Nvidia TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)
- [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
- [TensorRT python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
- [TensorRT c++ plugin API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin.html)
