# Code Structure of CUDA operators

This folder contains all non-python code for MMCV custom ops. Please follow the same architecture if you want to add new ops.

## Directories Tree

```folder
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

## Components

- `common`: This directory contains all tools and shared codes.
  - `cuda`: The cuda kernels which can be shared by all backends. **HIP** kernel is also here since they have similar syntax.
- `onnxruntime`: **ONNX Runtime** support for custom ops.
  - `cpu`: CPU implementation of supported ops.
- `parrots`: **Parrots** is a deep learning frame for model training and inference. Parrots custom ops are placed in this directory.
- `pytorch`: **PyTorch** custom ops are supported by binding C++ to Python with **pybind11**. The ops implementation and binding codes are placed in this directory.
  - `cuda`: This directory contains cuda kernel launchers, which feed memory pointers of tensor to the cuda kernel in `common/cuda`. The launchers provide c++ interface of cuda implementation of corresponding custom ops.
- `tensorrt`: **TensorRT** support for custom ops.
  - `plugins`: This directory contains the implementation of the supported custom ops. Some ops might also use shared cuda kernel in `common/cuda`.

## How to add new PyTorch ops?

1. (Optional) Add shared kernel in `common` to support special hardware platform.

    ```c++
    // src/common/cuda/new_ops_cuda_kernel.cuh

    template <typename T>
    __global__ void new_ops_forward_cuda_kernel(const T* input, T* output, ...) {
        // forward here
    }

    ```

    Add cuda kernel launcher in `pytorch/cuda`.

    ```c++
    // src/pytorch/cuda
    #include <new_ops_cuda_kernel.cuh>

    void NewOpsForwardCUDAKernelLauncher(Tensor input, Tensor output, ...){
        // initialize
        at::cuda::CUDAGuard device_guard(input.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ...
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "new_ops_forward_cuda_kernel", ([&] {
                new_ops_forward_cuda_kernel<scalar_t>
                    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),...);
            }));
        AT_CUDA_CHECK(cudaGetLastError());
    }
    ```

2. Add ops implementation in `pytorch` directory. Select different implementations according to device type.

    ```c++
    // src/pytorch/new_ops.cpp
    #ifdef MMCV_WITH_CUDA
    Tensor new_ops_forward_cuda(Tensor input, Tensor output, ...){
        // implement cuda forward here
        // use `NewOpsForwardCUDAKernelLauncher` here
    }
    #else

    Tensor new_ops_forward_cpu(Tensor input, Tensor output, ...){
        // implement cpu forward here
    }

    ...

    Tensor new_ops_forward(Tensor input, Tensor output, ...){
        // select implementation by input device type
        if (boxes.device().is_cuda()) {
        #ifdef MMCV_WITH_CUDA
            CHECK_CUDA_INPUT(input);
            CHECK_CUDA_INPUT(output);
            return new_ops_forward_cuda(input, output, ...);
        #else
            AT_ERROR("new ops is not compiled with GPU support");
        #endif
        } else {
            CHECK_CPU_INPUT(input);
            CHECK_CPU_INPUT(output);
            return new_ops_forward_cpu(input, output, ...);
        }
    }
    ```

3. Binding the implementation in `pytorch/pybind.cpp`

    ```c++
    // src/pytorch/pybind.cpp

    ...

    Tensor new_ops_forward(Tensor input, Tensor output, ...);

    ...

    // bind with pybind11
    m.def("new_ops_forward", &new_ops_forward, "new_ops_forward",
            py::arg("input"), py::arg("output"), ...);

    ...

    ```

4. Build MMCV again. Enjoy new ops in python

    ```python
    from ..utils import ext_loader
    ext_module = ext_loader.load_ext('_ext', ['new_ops_forward'])

    ...

    ext_module.new_ops_forward(input, output, ...)

    ```
