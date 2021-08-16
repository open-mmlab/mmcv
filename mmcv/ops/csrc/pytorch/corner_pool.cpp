// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/princeton-vl/CornerNet-Lite/tree/master/core/models/py_utils/_cpools/src
#include "pytorch_cpp_helper.hpp"

Tensor bottom_pool_forward(Tensor input) {
  // Initialize output
  Tensor output = at::zeros_like(input);
  // Get height
  int64_t height = input.size(2);
  output.copy_(input);

  for (int64_t ind = 1; ind < height; ind <<= 1) {
    Tensor max_temp = at::slice(output, 2, ind, height);
    Tensor cur_temp = at::slice(output, 2, ind, height).clone();
    Tensor next_temp = at::slice(output, 2, 0, height - ind).clone();
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

Tensor bottom_pool_backward(Tensor input, Tensor grad_output) {
  auto output = at::zeros_like(input);

  int32_t batch = input.size(0);
  int32_t channel = input.size(1);
  int32_t height = input.size(2);
  int32_t width = input.size(3);

  auto max_val = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kFloat));
  auto max_ind = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kLong));

  auto input_temp = input.select(2, 0);
  max_val.copy_(input_temp);

  max_ind.fill_(0);

  auto output_temp = output.select(2, 0);
  auto grad_output_temp = grad_output.select(2, 0);
  output_temp.copy_(grad_output_temp);

  auto un_max_ind = max_ind.unsqueeze(2);
  auto gt_mask = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kBool));
  auto max_temp = torch::zeros({batch, channel, width},
                               at::device(at::kCUDA).dtype(at::kFloat));
  for (int32_t ind = 0; ind < height - 1; ++ind) {
    input_temp = input.select(2, ind + 1);
    at::gt_out(gt_mask, input_temp, max_val);

    at::masked_select_out(max_temp, input_temp, gt_mask);
    max_val.masked_scatter_(gt_mask, max_temp);
    max_ind.masked_fill_(gt_mask, ind + 1);

    grad_output_temp = grad_output.select(2, ind + 1).unsqueeze(2);
    output.scatter_add_(2, un_max_ind, grad_output_temp);
  }

  return output;
}

Tensor left_pool_forward(Tensor input) {
  // Initialize output
  Tensor output = at::zeros_like(input);
  // Get width
  int64_t width = input.size(3);
  output.copy_(input);

  for (int64_t ind = 1; ind < width; ind <<= 1) {
    Tensor max_temp = at::slice(output, 3, 0, width - ind);
    Tensor cur_temp = at::slice(output, 3, 0, width - ind).clone();
    Tensor next_temp = at::slice(output, 3, ind, width).clone();
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

Tensor left_pool_backward(Tensor input, Tensor grad_output) {
  auto output = at::zeros_like(input);

  int32_t batch = input.size(0);
  int32_t channel = input.size(1);
  int32_t height = input.size(2);
  int32_t width = input.size(3);

  auto max_val = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kFloat));
  auto max_ind = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kLong));

  auto input_temp = input.select(3, width - 1);
  max_val.copy_(input_temp);

  max_ind.fill_(width - 1);

  auto output_temp = output.select(3, width - 1);
  auto grad_output_temp = grad_output.select(3, width - 1);
  output_temp.copy_(grad_output_temp);

  auto un_max_ind = max_ind.unsqueeze(3);
  auto gt_mask = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kBool));
  auto max_temp = torch::zeros({batch, channel, height},
                               at::device(at::kCUDA).dtype(at::kFloat));
  for (int32_t ind = 1; ind < width; ++ind) {
    input_temp = input.select(3, width - ind - 1);
    at::gt_out(gt_mask, input_temp, max_val);

    at::masked_select_out(max_temp, input_temp, gt_mask);
    max_val.masked_scatter_(gt_mask, max_temp);
    max_ind.masked_fill_(gt_mask, width - ind - 1);

    grad_output_temp = grad_output.select(3, width - ind - 1).unsqueeze(3);
    output.scatter_add_(3, un_max_ind, grad_output_temp);
  }

  return output;
}

Tensor right_pool_forward(Tensor input) {
  // Initialize output
  Tensor output = at::zeros_like(input);
  // Get width
  int64_t width = input.size(3);
  output.copy_(input);

  for (int64_t ind = 1; ind < width; ind <<= 1) {
    Tensor max_temp = at::slice(output, 3, ind, width);
    Tensor cur_temp = at::slice(output, 3, ind, width).clone();
    Tensor next_temp = at::slice(output, 3, 0, width - ind).clone();
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

Tensor right_pool_backward(Tensor input, Tensor grad_output) {
  Tensor output = at::zeros_like(input);

  int32_t batch = input.size(0);
  int32_t channel = input.size(1);
  int32_t height = input.size(2);
  int32_t width = input.size(3);

  auto max_val = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kFloat));
  auto max_ind = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kLong));

  auto input_temp = input.select(3, 0);
  max_val.copy_(input_temp);

  max_ind.fill_(0);

  auto output_temp = output.select(3, 0);
  auto grad_output_temp = grad_output.select(3, 0);
  output_temp.copy_(grad_output_temp);

  auto un_max_ind = max_ind.unsqueeze(3);
  auto gt_mask = torch::zeros({batch, channel, height},
                              at::device(at::kCUDA).dtype(at::kBool));
  auto max_temp = torch::zeros({batch, channel, height},
                               at::device(at::kCUDA).dtype(at::kFloat));
  for (int32_t ind = 0; ind < width - 1; ++ind) {
    input_temp = input.select(3, ind + 1);
    at::gt_out(gt_mask, input_temp, max_val);

    at::masked_select_out(max_temp, input_temp, gt_mask);
    max_val.masked_scatter_(gt_mask, max_temp);
    max_ind.masked_fill_(gt_mask, ind + 1);

    grad_output_temp = grad_output.select(3, ind + 1).unsqueeze(3);
    output.scatter_add_(3, un_max_ind, grad_output_temp);
  }

  return output;
}

Tensor top_pool_forward(Tensor input) {
  // Initialize output
  Tensor output = at::zeros_like(input);
  // Get height
  int64_t height = input.size(2);
  output.copy_(input);

  for (int64_t ind = 1; ind < height; ind <<= 1) {
    Tensor max_temp = at::slice(output, 2, 0, height - ind);
    Tensor cur_temp = at::slice(output, 2, 0, height - ind).clone();
    Tensor next_temp = at::slice(output, 2, ind, height).clone();
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

Tensor top_pool_backward(Tensor input, Tensor grad_output) {
  auto output = at::zeros_like(input);

  int32_t batch = input.size(0);
  int32_t channel = input.size(1);
  int32_t height = input.size(2);
  int32_t width = input.size(3);

  auto max_val = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kFloat));
  auto max_ind = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kLong));

  auto input_temp = input.select(2, height - 1);
  max_val.copy_(input_temp);

  max_ind.fill_(height - 1);

  auto output_temp = output.select(2, height - 1);
  auto grad_output_temp = grad_output.select(2, height - 1);
  output_temp.copy_(grad_output_temp);

  auto un_max_ind = max_ind.unsqueeze(2);
  auto gt_mask = torch::zeros({batch, channel, width},
                              at::device(at::kCUDA).dtype(at::kBool));
  auto max_temp = torch::zeros({batch, channel, width},
                               at::device(at::kCUDA).dtype(at::kFloat));
  for (int32_t ind = 1; ind < height; ++ind) {
    input_temp = input.select(2, height - ind - 1);
    at::gt_out(gt_mask, input_temp, max_val);

    at::masked_select_out(max_temp, input_temp, gt_mask);
    max_val.masked_scatter_(gt_mask, max_temp);
    max_ind.masked_fill_(gt_mask, height - ind - 1);

    grad_output_temp = grad_output.select(2, height - ind - 1).unsqueeze(2);
    output.scatter_add_(2, un_max_ind, grad_output_temp);
  }

  return output;
}
