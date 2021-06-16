# Introduction of `onnx` module in MMCV (Experimental)

## register_extra_symbolics

Some extra symbolic functions need to be registered before exporting PyTorch model to ONNX.

### Example

```python
import mmcv
from mmcv.onnx import register_extra_symbolics

opset_version = 11
register_extra_symbolics(opset_version)
```

### FAQs

- None
