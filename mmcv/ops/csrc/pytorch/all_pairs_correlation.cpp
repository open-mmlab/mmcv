// Copyright (c) OpenMMLab. All rights reserved.
#include <iostream>
#include "pytorch_cuda_helper.hpp"
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA

void AllPairsCorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                                  Tensor output);

void AllPairsCorrelationBackwardCUDAKernelLauncher(Tensor grad_output,
                                                   Tensor input1, Tensor input2,
                                                   Tensor grad_input1,
                                                   Tensor grad_intput2);
void all_pairs_correlation_cuda_forward(Tensor input1, Tensor input2,
                                        Tensor output)
{
    AllPairsCorrelationForwardCUDAKernelLauncher(input1, input2, output);
}

void all_pairs_correlation_cuda_backward(Tensor grad_output,
                                         Tensor input1, Tensor input2,
                                         Tensor grad_input1,
                                         Tensor grad_intput2)
{
    AllPairsCorrelationBackwardCUDAKernelLauncher(grad_output,
                                                  input1, input2,
                                                  grad_input1,
                                                  grad_intput2);
}
#endif

void all_pairs_correlation_forward(Tensor input1, Tensor input2, Tensor output)
{
    if (input1.device().is_cuda() and input2.device().is_cuda())
    {
#ifdef MMCV_WITH_CUDA
        CHECK_CUDA_INPUT(input1);
        CHECK_CUDA_INPUT(input2);
        all_pairs_correlation_cuda_forward(input1, input2, output);
#else
        AT_ERROR("Correlation is not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("All-pairs correlation is not implemented on CPU");
    }
}

void all_pairs_correlation_backward(Tensor grad_output,
                                    Tensor input1, Tensor input2,
                                    Tensor grad_input1, Tensor grad_input2)
{
    if (input1.device().is_cuda() and input2.device().is_cuda())
    {
#ifdef MMCV_WITH_CUDA
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CUDA_INPUT(input1);
        CHECK_CUDA_INPUT(input2);
        all_pairs_correlation_cuda_backward(grad_output, input1, input2,
                                            grad_input1, grad_input2);

#else
        AT_ERROR("All-pairs correlation is not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("Correlation is not implemented on CPU");
    }
}
