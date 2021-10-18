// Copyright (c) 2021, SenseTime.
#ifndef PYTHON_COMPUTE_EXT_PAT_EXT_SIGMOID_FOCAL_LOSS_CSRC_PARROTS_BANG_INTERNAL_H_
#define PYTHON_COMPUTE_EXT_PAT_EXT_SIGMOID_FOCAL_LOSS_CSRC_PARROTS_BANG_INTERNAL_H_

#include <cnrt.h>

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
        cnrtQueue_t queue, cnrtDataType_t d_type, const void* input,
        const void* target, const void* weight, const int32_t N,
        const int32_t C, const float alpha, const float gamma, void* output);

#endif  //  PYTHON_COMPUTE_EXT_PAT_EXT_SIGMOID_FOCAL_LOSS_CSRC_PARROTS_BANG_INTERNAL_H_  //  NOLINT
