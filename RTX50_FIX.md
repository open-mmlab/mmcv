# Fix: mmcv 2.0.1 CUDA Extensions on RTX 50 Series (Blackwell) / CUDA 12.8 / Windows

## Problem

Building mmcv CUDA extensions from source fails on Windows with:

- **GPU:** NVIDIA RTX 50 series (Blackwell, sm_120)
- **CUDA:** 12.8
- **PyTorch:** >= 2.7 (tested with 2.11)
- **Compiler:** MSVC 2022

Error output:

```
torch/include/torch/csrc/dynamo/compiled_autograd.h(1143): error C2872: 'std': ambiguous symbol
```

All CUDA extension files that include PyTorch headers fail to compile, including spconv ops, deformable convolution, ROI align, etc.

## Root Cause

When `nvcc` (CUDA compiler) invokes MSVC as its host compiler on Windows, it does **not** define the `_WIN32` preprocessor macro. This causes PyTorch's `compiled_autograd.h` to take the wrong branch of a `#if defined(_WIN32)` guard:

```cpp
// compiled_autograd.h ~line 1121
#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
    // Correct branch: skips the problematic code entirely
    TORCH_CHECK_NOT_IMPLEMENTED(false, "...");
#else
    // Wrong branch: contains ::std::is_same_v<T, ::std::string>
    // which triggers namespace ambiguity between ::std and cuda::std
    ...
    } else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    }
#endif
```

CUDA 12.8 headers introduce `cuda::std` namespace, which collides with `::std` when MSVC's non-standard preprocessor handles the code path. This is a known PyTorch bug tracked in:

- [pytorch/pytorch#173232](https://github.com/pytorch/pytorch/issues/173232)
- [pytorch/pytorch#148317](https://github.com/pytorch/pytorch/issues/148317)

## Fix

Three changes in `setup.py`:

### 1. Force `_WIN32` and `USE_CUDA` defines in nvcc flags

```python
if platform.system() == 'Windows':
    extra_compile_args['nvcc'] += [
        '-D_WIN32=1',
        '-DUSE_CUDA=1',
        '-Xcompiler=/Zc:preprocessor',
    ]
```

This makes `compiled_autograd.h` take the correct `#if defined(_WIN32)` branch, skipping the problematic `::std::is_same_v` code entirely.

### 2. Upgrade C++ standard from C++14 to C++17

```python
# Was: extra_compile_args['cxx'] = ['/std:c++14']
extra_compile_args['cxx'] = ['/std:c++17']
```

PyTorch >= 2.0 headers require C++17 features (`std::optional`, `std::string_view`, nested namespace definitions).

### 3. Replace deprecated `pkg_resources`

Replaced `from pkg_resources import ...` with `importlib.metadata` and `packaging.version`, since `pkg_resources` was removed in setuptools >= 82.

## Build

```bash
# From mmcv source root
pip wheel . --no-build-isolation -w dist/
# Result: dist/mmcv-2.0.1-cp310-cp310-win_amd64.whl

# Install
pip install dist/mmcv-2.0.1-cp310-cp310-win_amd64.whl
```

## Tested Configuration

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| GPU | RTX 50 series (Blackwell sm_120) |
| CUDA | 12.8 |
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| MSVC | 2022 BuildTools (19.44) |
| mmcv | 2.0.1 |

## Related Issues

- [pytorch/pytorch#173232](https://github.com/pytorch/pytorch/issues/173232) - compiled_autograd.h C2872
- [pytorch/pytorch#148317](https://github.com/pytorch/pytorch/issues/148317) - SageAttention C2872
- [pytorch/pytorch#166123](https://github.com/pytorch/pytorch/issues/166123) - torch-cublas-hgemm C2872
- [traveller59/spconv#746](https://github.com/traveller59/spconv/issues/746) - spconv on Blackwell cu128
- [SageAttention fix](https://github.com/woct0rdho/SageAttention/commit/914fe0ed304c3db4a27735d394ae23c5d5e79701) - same nvcc flag workaround
