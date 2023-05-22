### v2.0.0rc1

OpenMMLab 团队于 2022 年 9 月 1 日在世界人工智能大会发布了新一代训练引擎 [MMEngine](https://github.com/open-mmlab/mmengine)，它是一个用于训练深度学习模型的基础库。相比于 MMCV，它提供了更高级且通用的训练器、接口更加统一的开放架构以及可定制化程度更高的训练流程。

与此同时，MMCV 发布了 [2.x](https://github.com/open-mmlab/mmcv/tree/2.x) 预发布版本并将于 2023 年 1 月 1 日发布 2.x 正式版本。在 2.x 版本中，它有以下变化：

（1）删除了以下组件：

- `mmcv.fileio` 模块，删除于 PR [#2179](https://github.com/open-mmlab/mmcv/pull/2179)。在需要使用 FileIO 的地方使用 mmengine 中的 FileIO 模块
- `mmcv.runner`、`mmcv.parallel`、`mmcv.engine` 和 `mmcv.device`，删除于 PR [#2216](https://github.com/open-mmlab/mmcv/pull/2216)
- `mmcv.utils` 的所有类（例如 `Config` 和 `Registry`）和大部分函数，删除于 PR [#2217](https://github.com/open-mmlab/mmcv/pull/2217)，只保留少数和 mmcv 相关的函数
- `mmcv.onnx`、`mmcv.tensorrt` 模块以及相关的函数，删除于 PR [#2225](https://github.com/open-mmlab/mmcv/pull/2225)

（2）新增了 [`mmcv.transforms`](https://github.com/open-mmlab/mmcv/tree/2.x/mmcv/transforms) 数据变换模块

（3）在 PR [#2235](https://github.com/open-mmlab/mmcv/pull/2235) 中将包名 **mmcv** 重命名为 **mmcv-lite**、 **mmcv-full** 重命名为 **mmcv**。此外，将环境变量 `MMCV_WITH_OPS` 的默认值从 0 改为 1

<table class="docutils">
<thead>
  <tr>
    <th align="center">MMCV < 2.0</th>
    <th align="center">MMCV >= 2.0 </th>
<tbody>
  <tr>
  <td valign="top">

```bash
# 包含算子，因为 mmcv-full 的最高版本小于 2.0.0，所以无需加版本限制
pip install mmcv-full -f xxxx

# 不包含算子
pip install "mmcv < 2.0.0"
```

</td>
  <td valign="top">

```bash
# 包含算子
pip install "mmcv>=2.0.0rc1" -f xxxx

# 不包含算子，因为 mmcv-lite 的起始版本为 2.0.0rc1，所以无需加版本限制
pip install mmcv-lite
```

</td>
</tr>
</thead>
</table>

### v1.3.18

部分自定义算子对于不同的设备有不同实现，为此添加的大量宏命令与类型检查使得代码变得难以维护。例如：

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

为此我们设计了注册与分发的机制以更好的管理这些算子实现。

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

// 注册算子的cuda实现
void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);
REGISTER_DEVICE_IMPL(roi_align_forward_impl, CUDA, roi_align_forward_cuda);

// roi_align.cpp
// 使用dispatcher根据参数中的Tensor device类型对实现进行分发
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

为了灵活地支持更多的后端和硬件，例如 `NVIDIA GPUs` 、`AMD GPUs`，我们重构了 `mmcv/ops/csrc` 目录。注意，这次重构不会影响 API 的使用。更多相关信息，请参考 [PR1206](https://github.com/open-mmlab/mmcv/pull/1206)。

原始的目录结构如下所示

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

重构之后，它的结构如下所示

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
