// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src

#include "parrots_cuda_helper.hpp"
#include "psamask_cuda_kernel.cuh"

void PSAMaskForwardCUDAKernelLauncher(const int psa_type,
                                      const DArrayLite input, DArrayLite output,
                                      const int num_, const int h_feature,
                                      const int w_feature, const int h_mask,
                                      const int w_mask, const int half_h_mask,
                                      const int half_w_mask, CudaContext& ctx) {
  int nthreads = num_ * h_feature * w_feature;
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  if (psa_type == 0)
    PARROTS_DISPATCH_FLOATING_TYPES(input.elemType().prim(), [&] {
      psamask_collect_forward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
          nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
          half_w_mask, input.ptr<scalar_t>(), output.ptr<scalar_t>());
    });
  else
    PARROTS_DISPATCH_FLOATING_TYPES(input.elemType().prim(), [&] {
      psamask_distribute_forward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
          nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
          half_w_mask, input.ptr<scalar_t>(), output.ptr<scalar_t>());
    });
}

void PSAMaskBackwardCUDAKernelLauncher(
    const int psa_type, const DArrayLite grad_output, DArrayLite grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask,
    CudaContext& ctx) {
  int nthreads = num_ * h_feature * w_feature;
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  if (psa_type == 0)
    PARROTS_DISPATCH_FLOATING_TYPES(grad_input.elemType().prim(), [&] {
      psamask_collect_backward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
          nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
          half_w_mask, grad_output.ptr<scalar_t>(), grad_input.ptr<scalar_t>());
    });
  else
    PARROTS_DISPATCH_FLOATING_TYPES(grad_input.elemType().prim(), [&] {
      psamask_distribute_backward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
          nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
          half_w_mask, grad_output.ptr<scalar_t>(), grad_input.ptr<scalar_t>());
    });
}
