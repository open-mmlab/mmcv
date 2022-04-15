## Introduction of mmcv.onnx module

### <span style="color:red">DeprecationWarning</span>

ONNX support will be deprecated in the future.
Welcome to use the unified model deployment toolbox MMDeploy: https://github.com/open-mmlab/mmdeploy

### register_extra_symbolics

Some extra symbolic functions need to be registered before exporting PyTorch model to ONNX.

#### Example

```python
import mmcv
from mmcv.onnx import register_extra_symbolics

opset_version = 11
register_extra_symbolics(opset_version)
```

#### Reminder

- *Please note that this feature is experimental and may change in the future.*

#### FAQs

- None
