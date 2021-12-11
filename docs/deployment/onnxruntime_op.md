## ONNX Runtime Deployment

### Introduction of ONNX Runtime

**ONNX Runtime** is a cross-platform inferencing and training accelerator compatible with many popular ML/DNN frameworks. Check its [github](https://github.com/microsoft/onnxruntime) for more information.

### Introduction of ONNX

**ONNX** stands for **Open Neural Network Exchange**, which acts as *Intermediate Representation(IR)* for ML/DNN models from many frameworks. Check its [github](https://github.com/onnx/onnx) for more information.

### Why include custom operators for ONNX Runtime in MMCV

- To verify the correctness of exported ONNX models in ONNX Runtime.
- To ease the deployment of ONNX models with custom operators from `mmcv.ops` in ONNX Runtime.

### List of operators for ONNX Runtime supported in MMCV

| Operator                                               | CPU | GPU | MMCV Releases |
|:-------------------------------------------------------|:---:|:---:|:-------------:|
| [SoftNMS](onnxruntime_custom_ops.md#softnms)           |  Y  |  N  |     1.2.3     |
| [RoIAlign](onnxruntime_custom_ops.md#roialign)         |  Y  |  N  |     1.2.5     |
| [NMS](onnxruntime_custom_ops.md#nms)                   |  Y  |  N  |     1.2.7     |
| [grid_sampler](onnxruntime_custom_ops.md#grid_sampler) |  Y  |  N  |     1.3.1     |
| [CornerPool](onnxruntime_custom_ops.md#cornerpool)     |  Y  |  N  |     1.3.4     |
| [cummax](onnxruntime_custom_ops.md#cummax)             |  Y  |  N  |     1.3.4     |
| [cummin](onnxruntime_custom_ops.md#cummin)             |  Y  |  N  |     1.3.4     |

### How to build custom operators for ONNX Runtime

*Please be noted that only **onnxruntime>=1.8.1** of CPU version on Linux platform is tested by now.*

#### Prerequisite

- Clone repository

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

- Download `onnxruntime-linux` from ONNX Runtime [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1), extract it, expose `ONNXRUNTIME_DIR` and finally add the lib path to `LD_LIBRARY_PATH` as below:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

#### Build on Linux

```bash
cd mmcv ## to MMCV root directory
MMCV_WITH_OPS=1 MMCV_WITH_ORT=1 python setup.py develop
```

### How to do inference using exported ONNX models with custom operators in ONNX Runtime in python

Install ONNX Runtime with `pip`

```bash
pip install onnxruntime==1.8.1
```

Inference Demo

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

### How to add a new custom operator for ONNX Runtime in MMCV

#### Reminder

- *Please note that this feature is experimental and may change in the future. Strongly suggest users always try with the latest master branch.*

- The custom operator is not included in [supported operator list](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md) in ONNX Runtime.
- The custom operator should be able to be exported to ONNX.

#### Main procedures

Take custom operator `soft_nms` for example.

1. Add header `soft_nms.h` to ONNX Runtime include directory `mmcv/ops/csrc/onnxruntime/`
2. Add source `soft_nms.cpp` to ONNX Runtime source directory `mmcv/ops/csrc/onnxruntime/cpu/`
3. Register `soft_nms` operator in [onnxruntime_register.cpp](../../mmcv/ops/csrc/onnxruntime/cpu/onnxruntime_register.cpp)

    ```c++
    #include "soft_nms.h"

    SoftNmsOp c_SoftNmsOp;

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
    }
    ```

4. Add unit test into `tests/test_ops/test_onnx.py`
   Check [here](../../tests/test_ops/test_onnx.py) for examples.

**Finally, welcome to send us PR of adding custom operators for ONNX Runtime in MMCV.** :nerd_face:

### Known Issues

- "RuntimeError: tuple appears in op that does not forward tuples, unsupported kind: `prim::PythonOp`."
   1. Note generally `cummax` or `cummin` is exportable to ONNX as long as the torch version >= 1.5.0, since `torch.cummax` is only supported with torch >= 1.5.0. But when `cummax` or `cummin` serves as an intermediate component whose outputs is used as inputs for another modules, it's expected that torch version must be >= 1.7.0. Otherwise the above error might arise, when running exported ONNX model with onnxruntime.
   2. Solution: update the torch version to 1.7.0 or higher.

### References

- [How to export Pytorch model with custom op to ONNX and run it in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [How to add a custom operator/kernel in ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)
