#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void CAForwardCUDAKernelLauncher(const Tensor t, const Tensor f, Tensor weight);

void CABackwardCUDAKernelLauncher(const Tensor dw, const Tensor t,
                                  const Tensor f, Tensor dt, Tensor df);

void CAMapForwardCUDAKernelLauncher(const Tensor weight, const Tensor g,
                                    Tensor out);

void CAMapBackwardCUDAKernelLauncher(const Tensor dout, const Tensor weight,
                                     const Tensor g, Tensor dw, Tensor dg);

void ca_forward_cuda(const Tensor t, const Tensor f, Tensor weight) {
  CAForwardCUDAKernelLauncher(t, f, weight);
}

void ca_backward_cuda(const Tensor dw, const Tensor t, const Tensor f,
                      Tensor dt, Tensor df) {
  CABackwardCUDAKernelLauncher(dw, t, f, dt, df);
}

void ca_map_forward_cuda(const Tensor weight, const Tensor g, Tensor out) {
  CAMapForwardCUDAKernelLauncher(weight, g, out);
}

void ca_map_backward_cuda(const Tensor dout, const Tensor weight,
                          const Tensor g, Tensor dw, Tensor dg) {
  CAMapBackwardCUDAKernelLauncher(dout, weight, g, dw, dg);
}
#endif

void ca_forward(const Tensor t, const Tensor f, Tensor weight) {
  if (t.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(t);
    CHECK_CUDA_INPUT(f);
    CHECK_CUDA_INPUT(weight);
    ca_forward_cuda(t, f, weight);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ca is not implemented on the CPU");
  }
}

void ca_backward(const Tensor dw, const Tensor t, const Tensor f, Tensor dt,
                 Tensor df) {
  if (dw.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(dw);
    CHECK_CUDA_INPUT(t);
    CHECK_CUDA_INPUT(f);
    CHECK_CUDA_INPUT(dt);
    CHECK_CUDA_INPUT(df);
    ca_backward_cuda(dw, t, f, dt, df);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ca is not implemented on the CPU");
  }
}

void ca_map_forward(const Tensor weight, const Tensor g, Tensor out) {
  if (weight.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(g);
    CHECK_CUDA_INPUT(out);
    ca_map_forward_cuda(weight, g, out);
#else
    AT_ERROR("ca_map is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ca is not implemented on the CPU");
  }
}

void ca_map_backward(const Tensor dout, const Tensor weight, const Tensor g,
                     Tensor dw, Tensor dg) {
  if (dout.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(dout);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(g);
    CHECK_CUDA_INPUT(dw);
    CHECK_CUDA_INPUT(dg);
    ca_map_backward_cuda(dout, weight, g, dw, dg);
#else
    AT_ERROR("ca_map is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ca is not implemented on the CPU");
  }
}
