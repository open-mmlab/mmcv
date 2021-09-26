## MMCV中ONNX模块简介 (实验性)

### register_extra_symbolics

在将PyTorch模型导出成ONNX时，需要注册额外的符号函数

#### 范例

```python
import mmcv
from mmcv.onnx import register_extra_symbolics

opset_version = 11
register_extra_symbolics(opset_version)
```

#### 常见问题

- 无
