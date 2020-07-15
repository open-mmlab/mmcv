#include "parrots_cuda_helper.hpp"
#include "sigmoid_focal_loss_cuda_kernel.cuh"
#include "softmax_focal_loss_cuda_kernel.cuh"

void SigmoidFocalLossForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite output, float gamma, float alpha, cudaStream_t stream) {
  int output_size = output.size();
  int num_classes = input.dim(1);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.ptr<scalar_t>(), target.ptr<int64_t>(),
                weight.ptr<scalar_t>(), output.ptr<scalar_t>(), gamma, alpha,
                num_classes);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite grad_input, float gamma, float alpha, cudaStream_t stream) {
  int output_size = grad_input.size();
  int num_classes = input.dim(1);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.ptr<scalar_t>(), target.ptr<int64_t>(),
                weight.ptr<scalar_t>(), grad_input.ptr<scalar_t>(), gamma,
                alpha, num_classes);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossForwardCUDAKernelLauncher(
    const DArrayLite softmax, const DArrayLite target, const DArrayLite weight,
    DArrayLite output, float gamma, float alpha, cudaStream_t stream) {
  int output_size = output.size();
  int num_classes = softmax.dim(1);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      softmax.elemType().prim(), ([&] {
        softmax_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.ptr<scalar_t>(), target.ptr<int64_t>(),
                weight.ptr<scalar_t>(), output.ptr<scalar_t>(), gamma, alpha,
                num_classes);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossBackwardCUDAKernelLauncher(
    const DArrayLite softmax, const DArrayLite target, const DArrayLite weight,
    DArrayLite buff, DArrayLite grad_input, float gamma, float alpha,
    cudaStream_t stream) {
  int output_size = buff.size();
  int num_classes = softmax.dim(1);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.elemType().prim(), ([&] {
        softmax_focal_loss_backward_cuda1_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.ptr<scalar_t>(), target.ptr<int64_t>(),
                weight.ptr<scalar_t>(), buff.ptr<scalar_t>(), gamma, alpha,
                num_classes);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());

  output_size = grad_input.size();

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.elemType().prim(), ([&] {
        softmax_focal_loss_backward_cuda2_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.ptr<scalar_t>(), target.ptr<int64_t>(),
                buff.ptr<scalar_t>(), grad_input.ptr<scalar_t>(), num_classes);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
