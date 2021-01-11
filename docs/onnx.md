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

## ONNX simplify

### Intention

`mmcv.onnx.simplify` is based on [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), which is a useful tool to make exported ONNX models slimmer by performing a series of optimization. However, for Pytorch models with custom op from `mmcv`, it would break down. Thus, custom ops for ONNX Runtime should be registered.

### Prerequisite

`mmcv.onnx.simplify` has three dependencies: `onnx`, `onnxoptimizer`, `onnxruntime`. After installation of `mmcv`, you have to install them manually using pip.

```bash
pip install onnx onnxoptimizer onnxruntime
```

### Usage

```python
import onnx
import numpy as np

import mmcv
from mmcv.onnx.simplify import simplify

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
input = {'input':dummy_input}
input_file = 'sample.onnx'
output_file = 'slim.onnx'
model = simplify(input_file, [input], output_file)
```

### FAQs

- None
