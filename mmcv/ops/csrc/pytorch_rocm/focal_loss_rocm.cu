#include "hip/hip_runtime.h"
#include "pytorch_rocm_helper.hpp"
#include "sigmoid_focal_loss_cuda_kernel.cuh"
#include "softmax_focal_loss_cuda_kernel.cuh"

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);
  AT_ASSERTM(target.max().item<long>() <= (long)num_classes,
             "target label should smaller or equal than num classes");
  at::hip::HIPGuard device_guard(input.device());
  hipStream_t stream = at::hip::getCurrentHIPStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_forward_cuda_kernel", [&] {
        hipLaunchKernelGGL(sigmoid_focal_loss_forward_cuda_kernel<scalar_t>, dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(hipGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  at::hip::HIPGuard device_guard(grad_input.device());
  hipStream_t stream = at::hip::getCurrentHIPStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_backward_cuda_kernel", [&] {
        hipLaunchKernelGGL(sigmoid_focal_loss_backward_cuda_kernel<scalar_t>, dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(hipGetLastError());
}

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = softmax.size(1);

  AT_ASSERTM(target.max().item<long>() <= (long)num_classes,
             "target label should smaller or equal than num classes");
  at::hip::HIPGuard device_guard(softmax.device());
  hipStream_t stream = at::hip::getCurrentHIPStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      softmax.scalar_type(), "softmax_focal_loss_forward_cuda_kernel", [&] {
        hipLaunchKernelGGL(softmax_focal_loss_forward_cuda_kernel<scalar_t>, dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(hipGetLastError());
}

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int num_classes = softmax.size(1);

  int output_size = buff.numel();
  at::hip::HIPGuard device_guard(grad_input.device());
  hipStream_t stream = at::hip::getCurrentHIPStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(), "softmax_focal_loss_backward_cuda1_kernel",
      [&] {
        hipLaunchKernelGGL(softmax_focal_loss_backward_cuda1_kernel<scalar_t>, dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                buff.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(hipGetLastError());

  output_size = grad_input.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(), "softmax_focal_loss_backward_cuda2_kernel",
      [&] {
        hipLaunchKernelGGL(softmax_focal_loss_backward_cuda2_kernel<scalar_t>, dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), buff.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), num_classes);
      });

  AT_CUDA_CHECK(hipGetLastError());
}
