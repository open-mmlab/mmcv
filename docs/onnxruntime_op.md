## Custom operators for onnxruntime in mmcv

### Introduction of onnxruntime    
**ONNX RUNTIME** is a cross-platform inferencing and training accelerator compatible with many popular ML/DNN frameworks. Check its [github](https://github.com/microsoft/onnxruntime) for more information.

### Introduction of onnx
**ONNX** stands for **Open Neural Network Exchange**, which acts as *IR(Intermediate Representation)* for ML/DNN models from many frameworks. Check its [github](https://github.com/onnx/onnx) for more information.

### Why include custom operators for onnxruntime in mmcv
- To support verify correctness and precision of exported onnx models in onnxruntime.
- To support deploy onnx models with custom operators from `mmcv.ops` in onnxruntime.

### Custom operator list for onnxruntime supported in mmcv
| Operator |  CPU  |  GPU  | Note  |
| :------: | :---: | :---: | :---: |
| SoftNMS  |   Y   |   N   | None  |


### How to build custom operators for onnxruntime
*Please be noted that only **onnxruntime>=1.5.1** of CPU version on Linux platform is tested by now.*

#### Prerequisite
1. Clone repository
```bash
git clone https://github.com/open-mmlab/mmcv.git
```
2. Download `onnxruntime-linux-x64-1.5.1.tgz` from onnxruntime [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.5.1), extract it, expose `ONNXRUNTIME_DIR` and finally add the lib path to `LD_LIBRARY_PATH`:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.5.1/onnxruntime-linux-x64-1.5.1.tgz

tar -zxvf onnxruntime-linux-x64-1.5.1.tgz
cd onnxruntime-linux-x64-1.5.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

#### Build on Linux
```bash
cd mmcv # to mmcv root directory
MMCV_WITH_OPS=1 MMCV_WITH_ORT=1 pip install -e .
```

### How to do inference using exported onnx models with custom operators in onnxruntime in python
Install onnxruntime with `pip`
```bash
pip install onnxruntime==1.5.1
``` 
Inference Demo

```python
import os

import numpy as np
import onnxruntime as ort

from mmcv.ops import get_onnxruntime_op_path

assert os.path.exists(ort_custom_op_path)
session_options = ort.SessionOptions()
session_options.register_custom_ops_library(ort_custom_op_path)
# exported onnx model with custom operators
onnx_file = 'sample.onnx'
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
sess = ort.InferenceSession(onnx_file, session_options)
onnx_results = sess.run(None, {'input' : input_data})
```

### How to add a new custom operator for onnxruntime in mmcv

#### Prerequisite
* The custom operator is not included in [supported operator list](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md) in onnxruntime.
* The custom operator should be able to be exported to onnx.

#### Main procedures
Take custom operator `soft_nms` for example.
1. Add header `soft_nms.h` to onnxruntime include directory `mmcv/ops/csrc/onnxruntime/`
2. Add source `soft_nms.cpp` to onnxruntime source directory `mmcv/ops/csrc/onnxruntime/cpu/`
3. Register `soft_nms` operator in [onnxruntime_register.cpp](../mmcv/ops/csrc/onnxruntime/cpu/onnxruntime_register.cpp)

    ```c++
    #include "soft_nms.h"

    SoftNmsOp c_SoftNmsOp;

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
    }
    ```
4. Add unit test into `tests/test_ops/test_onnx.py`
   Check [here](../tests/test_ops/test_onnx.py) for examples.

**Finally, welcome to send us PR of adding custom operators for onnxruntime in mmcv.** :nerd_face:

### Known Issues
* None


### References
- [How to export Pytorch model with custom op to ONNX and run it in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [How to add a custom operator/kernel in onnxruntime](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)

