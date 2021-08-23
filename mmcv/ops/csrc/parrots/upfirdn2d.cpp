// Copyright (c) OpenMMLab. All rights reserved
// from
// https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
torch::Tensor upfirdn2d_op(const torch::Tensor &input,
                           const torch::Tensor &kernel, int up_x, int up_y,
                           int down_x, int down_y, int pad_x0, int pad_x1,
                           int pad_y0, int pad_y1);

#endif

torch::Tensor upfirdn2d(const torch::Tensor &input, const torch::Tensor &kernel,
                        int up_x, int up_y, int down_x, int down_y, int pad_x0,
                        int pad_x1, int pad_y0, int pad_y1) {
#ifdef MMCV_WITH_CUDA
  CHECK_CUDA(input);
  CHECK_CUDA(kernel);

  return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
                      pad_y0, pad_y1);
#else
  AT_ERROR("UpFirDn2d is not compiled with GPU support");
#endif
}
