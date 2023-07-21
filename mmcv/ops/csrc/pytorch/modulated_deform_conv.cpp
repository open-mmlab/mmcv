// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/base_def.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
#endif

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  DISPATCH_DEVICE_IMPL(modulated_deformable_im2col_impl, data_im, data_offset,
                       data_mask, batch_size, channels, height_im, width_im,
                       height_col, width_col, kernel_h, kernel_w, pad_h, pad_w,
                       stride_h, stride_w, dilation_h, dilation_w,
                       deformable_group, data_col);
}

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {
  DISPATCH_DEVICE_IMPL(modulated_deformable_col2im_impl, data_col, data_offset,
                       data_mask, batch_size, channels, height_im, width_im,
                       height_col, width_col, kernel_h, kernel_w, pad_h, pad_w,
                       stride_h, stride_w, dilation_h, dilation_w,
                       deformable_group, grad_im);
}

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask) {
  DISPATCH_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, data_col,
                       data_im, data_offset, data_mask, batch_size, channels,
                       height_im, width_im, height_col, width_col, kernel_h,
                       kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                       dilation_w, deformable_group, grad_offset, grad_mask);
}

void modulated_deform_conv_forward_fallthrough(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns =
      at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out},
                input.options());

  output = output.view({output.size(0), group, output.size(1) / group,
                        output.size(2), output.size(3)});

  for (int b = 0; b < batch; b++) {
    modulated_deformable_im2col_impl(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    // divide into group
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output = output.view({output.size(0), output.size(1) * output.size(2),
                        output.size(3), output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}

void modulated_deform_conv_backward_fallthrough(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out},
                      input.options());

  grad_output =
      grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    for (int g = 0; g < group; g++) {
      columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                        grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_impl(
        columns, input[b], offset[b], mask[b], 1, channels, height, width,
        height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, deformable_group, grad_offset[b],
        grad_mask[b]);
    // gradient w.r.t. input data
    modulated_deformable_col2im_impl(
        columns, offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, grad_input[b]);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_impl(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
              .view_as(grad_weight[g]);
      if (with_bias) {
        grad_bias[g] =
            grad_bias[g]
                .view({-1, 1})
                .addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}))
                .view(-1);
      }
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                    grad_weight.size(2), grad_weight.size(3),
                                    grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view({grad_output.size(0) * grad_output.size(1),
                                  grad_output.size(2), grad_output.size(3),
                                  grad_output.size(4)});
}

#ifdef MMCV_WITH_DIOPI
void modulated_deform_conv_forward_diopi(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
    modulated_deform_conv_forward_fallthrough(
        input, weight, bias, ones, offset, mask, output, columns, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto weight_p = toDiopiTensorHandle(weight);
  auto bias_p = toDiopiTensorHandle(bias);
  auto ones_p = toDiopiTensorHandle(ones);
  auto offset_p = toDiopiTensorHandle(offset);
  auto mask_p = toDiopiTensorHandle(mask);
  auto output_p = toDiopiTensorHandle(output);
  auto columns_p = toDiopiTensorHandle(columns);
  bool is_mock_cuda = input.device().type() == dipu::DIPU_DEVICE_TYPE;
  if (is_mock_cuda && reinterpret_cast<void*>(diopiModulatedDeformConvMmcv) != nullptr) {
    auto ret = diopiModulatedDeformConvMmcv(
        ch, output_p, columns_p, ones_p, input_p, weight_p, bias_p, offset_p,
        mask_p, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, group, deformable_group, with_bias);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op modulated_deform_conv_forward";
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto bias_cpu = bias.cpu();
  auto ones_cpu = ones.cpu();
  auto offset_cpu = offset.cpu();
  auto mask_cpu = mask.cpu();
  auto output_cpu = output.cpu();
  auto columns_cpu = columns.cpu();
  modulated_deform_conv_forward_fallthrough(
      input_cpu, weight_cpu, bias_cpu, ones_cpu, offset_cpu, mask_cpu,
      output_cpu, columns_cpu, kernel_h, kernel_w, stride_h, stride_w, pad_h,
      pad_w, dilation_h, dilation_w, group, deformable_group, with_bias);
  output.copy_(output_cpu);
  return;
}

void modulated_deform_conv_backward_diopi(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
    modulated_deform_conv_backward_fallthrough(
        input, weight, bias, ones, offset, mask, columns, grad_input,
        grad_weight, grad_bias, grad_offset, grad_mask, grad_output, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto weight_p = toDiopiTensorHandle(weight);
  auto bias_p = toDiopiTensorHandle(bias);
  auto ones_p = toDiopiTensorHandle(ones);
  auto offset_p = toDiopiTensorHandle(offset);
  auto mask_p = toDiopiTensorHandle(mask);
  auto columns_p = toDiopiTensorHandle(columns);
  auto grad_input_p = toDiopiTensorHandle(grad_input);
  auto grad_weight_p = toDiopiTensorHandle(grad_weight);
  auto grad_bias_p = toDiopiTensorHandle(grad_bias);
  auto grad_offset_p = toDiopiTensorHandle(grad_offset);
  auto grad_mask_p = toDiopiTensorHandle(grad_mask);
  auto grad_output_p = toDiopiTensorHandle(grad_output);
  bool is_mock_cuda = input.device().type() == dipu::DIPU_DEVICE_TYPE;

  if (is_mock_cuda && reinterpret_cast<void*>(diopiModulatedDeformConvBackwardMmcv) !=
      nullptr) {
    auto ret = diopiModulatedDeformConvBackwardMmcv(
        ch, grad_input_p, grad_weight_p, grad_bias_p, grad_offset_p,
        grad_mask_p, input_p, weight_p, bias_p, ones_p, offset_p, mask_p,
        columns_p, grad_output_p, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group, deformable_group, with_bias);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op modulated_deform_conv_forward";
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto bias_cpu = bias.cpu();
  auto ones_cpu = ones.cpu();
  auto offset_cpu = offset.cpu();
  auto mask_cpu = mask.cpu();
  auto columns_cpu = columns.cpu();
  auto grad_input_cpu = grad_input.cpu();
  auto grad_weight_cpu = grad_weight.cpu();
  auto grad_bias_cpu = grad_bias.cpu();
  auto grad_offset_cpu = grad_offset.cpu();
  auto grad_mask_cpu = grad_mask.cpu();
  auto grad_output_cpu = grad_output.cpu();
  modulated_deform_conv_backward_fallthrough(
      input_cpu, weight_cpu, bias_cpu, ones_cpu, offset_cpu, mask_cpu,
      columns_cpu, grad_input_cpu, grad_weight_cpu, grad_bias_cpu,
      grad_offset_cpu, grad_mask_cpu, grad_output_cpu, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
  grad_input.copy_(grad_input_cpu);
  grad_weight.copy_(grad_weight_cpu);
  grad_bias.copy_(grad_bias_cpu);
  grad_offset.copy_(grad_offset_cpu);
  grad_mask.copy_(grad_mask_cpu);
  return;
}
#endif

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
#ifdef MMCV_WITH_DIOPI
  modulated_deform_conv_forward_diopi(
      input, weight, bias, ones, offset, mask, output, columns, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
#else
  modulated_deform_conv_forward_fallthrough(
      input, weight, bias, ones, offset, mask, output, columns, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
#endif
}

void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
#ifdef MMCV_WITH_DIOPI
  modulated_deform_conv_backward_diopi(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
#else
  modulated_deform_conv_backward_fallthrough(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
#endif
}
