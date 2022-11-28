
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <iostream>
#include <math.h>

#include "helper.hpp"
#include "cuda/cuda_kernel.h"

#define FLT_MIN		__FLT_MIN__

extern "C" {

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, const diopiTensorHandle_t dets,
        const diopiTensorHandle_t scores, double iouThreshold, int64_t offset) {
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = mmcv::diopiops::NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    
}

}  // extern "C"
