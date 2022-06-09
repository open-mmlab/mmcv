## MMCV中的ONNX Runtime自定义算子

### ONNX Runtime介绍

**ONNX Runtime**是一个跨平台的推理与训练加速器，适配许多常用的机器学习/深度神经网络框架。请访问[github](https://github.com/microsoft/onnxruntime)了解更多信息。

### ONNX介绍

**ONNX**是**Open Neural Network Exchange**的缩写，是许多机器学习/深度神经网络框架使用的*中间表示(IR)*。请访问[github](https://github.com/onnx/onnx)了解更多信息。

### 为什么要在MMCV中添加ONNX自定义算子？

- 为了验证ONNX模型在ONNX Runtime下的推理的正确性。
- 为了方便使用了`mmcv.ops`自定义算子的模型的部署工作。

### MMCV已支持的算子

|                                       算子                                       | CPU | GPU | MMCV版本 |
| :------------------------------------------------------------------------------: | :-: | :-: | :------: |
|                   [SoftNMS](onnxruntime_custom_ops.md#softnms)                   |  Y  |  N  |  1.2.3   |
|                  [RoIAlign](onnxruntime_custom_ops.md#roialign)                  |  Y  |  N  |  1.2.5   |
|                       [NMS](onnxruntime_custom_ops.md#nms)                       |  Y  |  N  |  1.2.7   |
|              [grid_sampler](onnxruntime_custom_ops.md#grid_sampler)              |  Y  |  N  |  1.3.1   |
|                [CornerPool](onnxruntime_custom_ops.md#cornerpool)                |  Y  |  N  |  1.3.4   |
|                    [cummax](onnxruntime_custom_ops.md#cummax)                    |  Y  |  N  |  1.3.4   |
|                    [cummin](onnxruntime_custom_ops.md#cummin)                    |  Y  |  N  |  1.3.4   |
| [MMCVModulatedDeformConv2d](onnxruntime_custom_ops.md#mmcvmodulateddeformconv2d) |  Y  |  N  |  1.3.12  |

### 如何编译ONNX Runtime自定义算子？

*请注意我们仅在**onnxruntime>=1.8.1**的Linux x86-64 cpu平台上进行过测试*

#### 准备工作

- 克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

- 从ONNX Runtime下载`onnxruntime-linux`：[releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1)，解压缩，根据路径创建变量`ONNXRUNTIME_DIR`并把路径下的lib目录添加到`LD_LIBRARY_PATH`，步骤如下：

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

#### Linux系统下编译

```bash
cd mmcv ## to MMCV root directory
MMCV_WITH_OPS=1 MMCV_WITH_ORT=1 python setup.py develop
```

### 如何在python下使用ONNX Runtime对导出的ONNX模型做编译

使用`pip`安装ONNX Runtime

```bash
pip install onnxruntime==1.8.1
```

推理范例

```python
import os

import numpy as np
import onnxruntime as ort

from mmcv.ops import get_onnxruntime_op_path

ort_custom_op_path = get_onnxruntime_op_path()
assert os.path.exists(ort_custom_op_path)
session_options = ort.SessionOptions()
session_options.register_custom_ops_library(ort_custom_op_path)
## exported ONNX model with custom operators
onnx_file = 'sample.onnx'
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
sess = ort.InferenceSession(onnx_file, session_options)
onnx_results = sess.run(None, {'input' : input_data})
```

### 如何为MMCV添加ONNX Runtime的自定义算子

#### 开发前提醒

- 该算子的ONNX Runtime实现尚未在MMCV中支持[已实现算子列表](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md)。
- 确保该自定义算子可以被ONNX导出。

#### 添加方法

以`soft_nms`为例：

1. 在ONNX Runtime头文件目录`mmcv/ops/csrc/onnxruntime/`下添加头文件`soft_nms.h`

2. 在ONNX Runtime源码目录`mmcv/ops/csrc/onnxruntime/cpu/`下添加算子实现`soft_nms.cpp`

3. 在[onnxruntime_register.cpp](../../../mmcv/ops/csrc/onnxruntime/cpu/onnxruntime_register.cpp)中注册实现的算子`soft_nms`

   ```c++
   #include "soft_nms.h"

   SoftNmsOp c_SoftNmsOp;

   if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
   return status;
   }
   ```

4. 在`tests/test_ops/test_onnx.py`添加单元测试，
   可以参考[here](../../tests/test_ops/test_onnx.py)。

**最后，欢迎为MMCV添加ONNX Runtime自定义算子** :nerd_face:

### 已知问题

- "RuntimeError: tuple appears in op that does not forward tuples, unsupported kind: `prim::PythonOp`."
  1. 请注意`cummax`和`cummin`算子是在torch >= 1.5.0被添加的。但他们需要在torch version >= 1.7.0才能正确导出。否则会在导出时发生上面的错误。
  2. 解决方法：升级PyTorch到1.7.0以上版本

### 引用

- [How to export Pytorch model with custom op to ONNX and run it in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [How to add a custom operator/kernel in ONNX Runtime](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
