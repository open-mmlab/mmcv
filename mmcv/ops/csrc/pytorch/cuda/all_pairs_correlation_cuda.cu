// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/princeton-vl/RAFT/blob/master/alt_cuda_corr/correlation_kernel.cu
// Original licence: Copyright (c) 2020, princeton-vl, under BSD 3-Clause License.

#include "all_pairs_correlation_cuda.cuh"

void AllPairsCorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
    Tensor output)
{
    const int batch_size = input1.size(0);
    const int iH1 = input1.size(2);
    const int iW1 = input1.size(3);
    const int iH2 = input2.size(2);
    const int iW2 = input2.size(2);

    auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
    auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();

    const int threads = THREADS_FORWARD;
    const dim3 blocks(batch_size, iH1, iW1);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(),
                                        "all_pairs_correlation_forward_cuda",
    (
        [&]{
            TensorAcc4R trInput1_acc = trInput1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
            TensorAcc4R trInput2_acc = trInput2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
            TensorAcc5R output_acc = output.packed_accessor32<scalar_t,5,RestrictPtrTraits>();
            all_pairs_correlation_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                trInput1_acc, trInput2_acc, output_acc);
        }
    ));

}

void AllPairsCorrelationBackwardCUDAKernelLauncher(Tensor grad_output,
    Tensor input1, Tensor input2,
    Tensor grad_input1,
    Tensor grad_input2)
{
    const int batch_size = input1.size(0);
    const int iH1 = input1.size(2);
    const int iW1 = input1.size(3);
    const int iH2 = input2.size(2);
    const int iW2 = input2.size(2);
    const int C = input1.size(1);

    const dim3 blocks1(C, iH1, iW1);
    const dim3 blocks2(C, iH2, iW2);
    const dim3 threads(THREADS_BACKWARD, THREADS_BACKWARD);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(),
                                        "all_pairs_correlation_backward_cuda",
    (
        [&]{
    TensorAcc4R input1_acc = input1.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
    TensorAcc4R input2_acc = input2.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
    TensorAcc4R grad_input1_acc = grad_input1.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
    TensorAcc4R grad_input2_acc = grad_input2.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
    TensorAcc5R grad_output_acc = grad_output.packed_accessor32<scalar_t,5,RestrictPtrTraits>();

    for (int n = 0; n < batch_size; ++n){
        all_pairs_correlation_backward_cuda_kernel_input1<scalar_t><<<blocks1, threads>>>(
            grad_output_acc, input2_acc, grad_input1_acc, n);
      }

      for (int n = 0; n < batch_size; ++n){
        all_pairs_correlation_backward_cuda_kernel_input2<scalar_t><<<blocks2, threads>>>(
            grad_output_acc, input1_acc, grad_input2_acc, n);

    }
    }));

}
