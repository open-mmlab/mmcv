// Copyright (c) OpenMMLab. All rights reserved
#include "deform_conv_utils.hpp"
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA

void deformable_im2col(Tensor data_im, Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       Tensor data_col);

void deformable_col2im(Tensor data_col, Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       Tensor grad_im);

void deformable_col2im_coord(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

#endif

void deformable_im2col_cpu(Tensor data_im, Tensor data_offset,
                           const int channels, const int height,
                           const int width, const int ksize_h,
                           const int ksize_w, const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w,
                           const int dilation_h, const int dilation_w,
                           const int parallel_imgs, const int deformable_group,
                           Tensor data_col);

void deformable_col2im_cpu(Tensor data_col, Tensor data_offset,
                           const int channels, const int height,
                           const int width, const int ksize_h,
                           const int ksize_w, const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w,
                           const int dilation_h, const int dilation_w,
                           const int parallel_imgs, const int deformable_group,
                           Tensor grad_im);

void deformable_col2im_coord_cpu(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

void deform_conv_forward(Tensor input, Tensor weight, Tensor offset,
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int dW, int dH, int padW, int padH,
                         int dilationW, int dilationH, int group,
                         int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(offset);
    CHECK_CPU_INPUT(weight);
    CHECK_CPU_INPUT(output);
    CHECK_CPU_INPUT(columns);
    CHECK_CPU_INPUT(ones);
  }

  deform_conv_shape_check(input, offset, NULL, weight, kH, kW, dH, dW, padH,
                          padW, dilationH, dilationW, group, deformable_group);
  at::DeviceGuard guard(input.device());

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane,
                        outputHeight, outputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  Tensor output_buffer = at::zeros({batchSize / im2col_step, nOutputPlane,
                                    im2col_step * outputHeight, outputWidth},
                                   output.options());

  output_buffer = output_buffer.view(
      {output_buffer.size(0), group, output_buffer.size(1) / group,
       output_buffer.size(2), output_buffer.size(3)});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
      deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                        dilationW, im2col_step, deformable_group, columns);
#endif
    } else {
      deformable_im2col_cpu(input[elt], offset[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                            dilationW, im2col_step, deformable_group, columns);
    }

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(weight[g].flatten(1), columns[g])
                                  .view_as(output_buffer[elt][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3), output_buffer.size(4)});

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                      im2col_step, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }
}

void deform_conv_backward_input(Tensor input, Tensor offset, Tensor gradOutput,
                                Tensor gradInput, Tensor gradOffset,
                                Tensor weight, Tensor columns, int kW, int kH,
                                int dW, int dH, int padW, int padH,
                                int dilationW, int dilationH, int group,
                                int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradInput);
    CHECK_CUDA_INPUT(gradOffset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(columns);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(offset);
    CHECK_CPU_INPUT(gradOutput);
    CHECK_CPU_INPUT(gradInput);
    CHECK_CPU_INPUT(gradOffset);
    CHECK_CPU_INPUT(weight);
    CHECK_CPU_INPUT(columns);
  }
  deform_conv_shape_check(input, offset, &gradOutput, weight, kH, kW, dH, dW,
                          padH, padW, dilationH, dilationW, group,
                          deformable_group);

  at::DeviceGuard guard(input.device());

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane,
                              inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step,
                                deformable_group * 2 * kH * kW, outputHeight,
                                outputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0), group, gradOutput.size(1) / group,
         gradOutput.size(2), gradOutput.size(3), gradOutput.size(4)});

    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                     gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2),
         gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});

    if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
      deformable_col2im_coord(columns, input[elt], offset[elt], nInputPlane,
                              inputHeight, inputWidth, kH, kW, padH, padW, dH,
                              dW, dilationH, dilationW, im2col_step,
                              deformable_group, gradOffset[elt]);

      deformable_col2im(columns, offset[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                        dilationW, im2col_step, deformable_group,
                        gradInput[elt]);
#endif
    } else {
      deformable_col2im_coord_cpu(columns, input[elt], offset[elt], nInputPlane,
                                  inputHeight, inputWidth, kH, kW, padH, padW,
                                  dH, dW, dilationH, dilationW, im2col_step,
                                  deformable_group, gradOffset[elt]);

      deformable_col2im_cpu(columns, offset[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                            dilationW, im2col_step, deformable_group,
                            gradInput[elt]);
    }

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
  }

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }
}

void deform_conv_backward_parameters(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int dW, int dH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     int deformable_group, float scale,
                                     int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradWeight);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(offset);
    CHECK_CPU_INPUT(gradOutput);
    CHECK_CPU_INPUT(gradWeight);
    CHECK_CPU_INPUT(columns);
    CHECK_CPU_INPUT(ones);
  }

  deform_conv_shape_check(input, offset, &gradOutput, gradWeight, kH, kW, dH,
                          dW, padH, padW, dilationH, dilationW, group,
                          deformable_group);
  at::DeviceGuard guard(input.device());

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view(
        at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step,
                             outputHeight, outputWidth});
  gradOutputBuffer = gradOutputBuffer.contiguous();
  gradOutputBuffer.copy_(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane,
                             im2col_step * outputHeight, outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {

    if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
      deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                        dilationW, im2col_step, deformable_group, columns);
#endif
    } else {
      deformable_im2col_cpu(input[elt], offset[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                            dilationW, im2col_step, deformable_group, columns);
    }

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group,
         gradOutputBuffer.size(2), gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight =
        gradWeight.view({group, gradWeight.size(0) / group, gradWeight.size(1),
                         gradWeight.size(2), gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .flatten(1)
                          .addmm_(gradOutputBuffer[elt][g].flatten(1),
                                  columns[g].transpose(1, 0), 1.0, scale)
                          .view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
         gradOutputBuffer.size(3), gradOutputBuffer.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1),
                                  gradWeight.size(2), gradWeight.size(3),
                                  gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }
}
