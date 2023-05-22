### v2.0.0rc1

The OpenMMLab team released a new generation of training engine [MMEngine](https://github.com/open-mmlab/mmengine) at the World Artificial Intelligence Conference on September 1, 2022. It is a foundational library for training deep learning models. Compared with MMCV, it provides a universal and powerful runner, an open architecture with a more unified interface, and a more customizable training process.

At the same time, MMCV released [2.x](https://github.com/open-mmlab/mmcv/tree/2.x) release candidate version and will release 2.x official version on January 1, 2023. In version 2.x, it has the following changes:

(1) It removed the following components:

- `mmcv.fileio` module, removed in PR [#2179](https://github.com/open-mmlab/mmcv/pull/2179). FileIO module from mmengine will be used wherever required.
- `mmcv.runner`, `mmcv.parallel`, `mmcv. engine` and `mmcv.device`, removed in PR [#2216](https://github.com/open-mmlab/mmcv/pull/2216).
- All classes in `mmcv.utils` (eg `Config` and `Registry`) and many functions, removed in PR [#2217](https://github.com/open-mmlab/mmcv/pull/2217). Only a few functions related to mmcv are reserved.
- `mmcv.onnx`, `mmcv.tensorrt` modules and related functions, removed in PR [#2225](https://github.com/open-mmlab/mmcv/pull/2225).

(2) It added the [`mmcv.transforms`](https://github.com/open-mmlab/mmcv/tree/2.x/mmcv/transforms) data transformation module.

(3) It renamed the package name **mmcv** to **mmcv-lite** and **mmcv-full** to **mmcv** in PR [#2235](https://github.com/open-mmlab/mmcv/pull/2235). Also, change the default value of the environment variable `MMCV_WITH_OPS` from 0 to 1.

<table class="docutils">
<thead>
  <tr>
    <th align="center">MMCV < 2.0</th>
    <th align="center">MMCV >= 2.0 </th>
<tbody>
  <tr>
  <td valign="top">

```bash
# Contains ops, because the highest version of mmcv-full is less than 2.0.0, so there is no need to add version restrictions
pip install mmcv-full -f xxxx

# do not contain ops
pip install "mmcv < 2.0.0"
```

</td>
  <td valign="top">

```bash
# Contains ops
pip install "mmcv>=2.0.0rc1" -f xxxx

# Ops are not included, because the starting version of mmcv-lite is 2.0.0rc1, so there is no need to add version restrictions
pip install mmcv-lite
```

</td>
</tr>
</thead>
</table>

### v1.3.18

Some ops have different implementations on different devices. Lots of macros and type checks are scattered in several files, which makes the code hard to maintain. For example:

```c++
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(argmax_y);
    CHECK_CUDA_INPUT(argmax_x);

    roi_align_forward_cuda(input, rois, output, argmax_y, argmax_x,
                           aligned_height, aligned_width, spatial_scale,
                           sampling_ratio, pool_mode, aligned);
#else
    AT_ERROR("RoIAlign is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(rois);
    CHECK_CPU_INPUT(output);
    CHECK_CPU_INPUT(argmax_y);
    CHECK_CPU_INPUT(argmax_x);
    roi_align_forward_cpu(input, rois, output, argmax_y, argmax_x,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
  }
```

Registry and dispatcher are added to manage these implementations.

```c++

void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned);

void roi_align_forward_cuda(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

// register cuda implementation
void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);
REGISTER_DEVICE_IMPL(roi_align_forward_impl, CUDA, roi_align_forward_cuda);

// roi_align.cpp
// use the dispatcher to invoke different implementation depending on device type of input tensors.
void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  DISPATCH_DEVICE_IMPL(roi_align_forward_impl, input, rois, output, argmax_y,
                       argmax_x, aligned_height, aligned_width, spatial_scale,
                       sampling_ratio, pool_mode, aligned);
}

```

### v1.3.11

In order to flexibly support more backends and hardwares like `NVIDIA GPUs` and `AMD GPUs`, the directory of `mmcv/ops/csrc` is refactored. Note that this refactoring will not affect the usage in API. For related information, please refer to [PR1206](https://github.com/open-mmlab/mmcv/pull/1206).

The original directory was organized as follows.

```
.
├── common_cuda_helper.hpp
├── ops_cuda_kernel.cuh
├── pytorch_cpp_helper.hpp
├── pytorch_cuda_helper.hpp
├── parrots_cpp_helper.hpp
├── parrots_cuda_helper.hpp
├── parrots_cudawarpfunction.cuh
├── onnxruntime
│   ├── onnxruntime_register.h
│   ├── onnxruntime_session_options_config_keys.h
│   ├── ort_mmcv_utils.h
│   ├── ...
│   ├── onnx_ops.h
│   └── cpu
│       ├── onnxruntime_register.cpp
│       ├── ...
│       └── onnx_ops_impl.cpp
├── parrots
│   ├── ...
│   ├── ops.cpp
│   ├── ops_cuda.cu
│   ├── ops_parrots.cpp
│   └── ops_pytorch.h
├── pytorch
│   ├── ...
│   ├── ops.cpp
│   ├── ops_cuda.cu
│   ├── pybind.cpp
└── tensorrt
    ├── trt_cuda_helper.cuh
    ├── trt_plugin_helper.hpp
    ├── trt_plugin.hpp
    ├── trt_serialize.hpp
    ├── ...
    ├── trt_ops.hpp
    └── plugins
        ├── trt_cuda_helper.cu
        ├── trt_plugin.cpp
        ├── ...
        ├── trt_ops.cpp
        └── trt_ops_kernel.cu
```

After refactored, it is organized as follows.

```
.
├── common
│   ├── box_iou_rotated_utils.hpp
│   ├── parrots_cpp_helper.hpp
│   ├── parrots_cuda_helper.hpp
│   ├── pytorch_cpp_helper.hpp
│   ├── pytorch_cuda_helper.hpp
│   └── cuda
│       ├── common_cuda_helper.hpp
│       ├── parrots_cudawarpfunction.cuh
│       ├── ...
│       └── ops_cuda_kernel.cuh
├── onnxruntime
│   ├── onnxruntime_register.h
│   ├── onnxruntime_session_options_config_keys.h
│   ├── ort_mmcv_utils.h
│   ├── ...
│   ├── onnx_ops.h
│   └── cpu
│       ├── onnxruntime_register.cpp
│       ├── ...
│       └── onnx_ops_impl.cpp
├── parrots
│   ├── ...
│   ├── ops.cpp
│   ├── ops_parrots.cpp
│   └── ops_pytorch.h
├── pytorch
│   ├── info.cpp
│   ├── pybind.cpp
│   ├── ...
│   ├── ops.cpp
│   └── cuda
│       ├── ...
│       └── ops_cuda.cu
└── tensorrt
    ├── trt_cuda_helper.cuh
    ├── trt_plugin_helper.hpp
    ├── trt_plugin.hpp
    ├── trt_serialize.hpp
    ├── ...
    ├── trt_ops.hpp
    └── plugins
        ├── trt_cuda_helper.cu
        ├── trt_plugin.cpp
        ├── ...
        ├── trt_ops.cpp
        └── trt_ops_kernel.cu
```
