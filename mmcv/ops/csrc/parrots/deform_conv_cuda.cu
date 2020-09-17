#include "deform_conv_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void deformable_im2col(DArrayLite data_im, DArrayLite data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       DArrayLite data_col, cudaStream_t stream) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.elemType().prim(), ([&] {
        deformable_im2col_gpu_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, data_im.ptr<scalar_t>(),
                data_offset.ptr<scalar_t>(), height, width, ksize_h, ksize_w,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                channel_per_deformable_group, parallel_imgs, channels,
                deformable_group, height_col, width_col,
                data_col.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void deformable_col2im(DArrayLite data_col, DArrayLite data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       DArrayLite grad_im, cudaStream_t stream) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.elemType().prim(), ([&] {
        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_col.ptr<scalar_t>(), data_offset.ptr<scalar_t>(),
            channels, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, deformable_group, height_col, width_col,
            grad_im.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void deformable_col2im_coord(
    DArrayLite data_col, DArrayLite data_im, DArrayLite data_offset,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, DArrayLite grad_offset,
    cudaStream_t stream) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w *
                    deformable_group * parallel_imgs;
  int channel_per_deformable_group =
      channels * ksize_h * ksize_w / deformable_group;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.elemType().prim(), ([&] {
        deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                             THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_col.ptr<scalar_t>(), data_im.ptr<scalar_t>(),
            data_offset.ptr<scalar_t>(), channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs,
            2 * ksize_h * ksize_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void deform_conv_shape_check(DArrayLite input, DArrayLite offset,
                             DArrayLite* gradOutput, DArrayLite weight, int kH,
                             int kW, int dH, int dW, int padH, int padW,
                             int dilationH, int dilationW, int group,
                             int deformable_group) {
  PARROTS_CHECKARGS(weight.ndims() == 4)
      << "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: "
      << weight.ndims();

  PARROTS_CHECKARGS(weight.isContiguous())
      << "weight tensor has to be contiguous";

  PARROTS_CHECKARGS(kW > 0 && kH > 0)
      << "kernel size should be greater than zero, but got kH: " << kH
      << " kW: " << kW;

  PARROTS_CHECKARGS(weight.dim(2) == kH && weight.dim(3) == kW)
      << "kernel size should be consistent with weight, but got kH: " << kH
      << " kW: " << kW << " weight.dim(2): " << weight.dim(2)
      << ", weight.dim(3): " << weight.dim(3);

  PARROTS_CHECKARGS(dW > 0 && dH > 0)
      << "stride should be greater than zero, but got dH: " << dH
      << " dW: " << dW;

  PARROTS_CHECKARGS(dilationW > 0 && dilationH > 0)
      << "dilation should be greater than 0, but got dilationH: " << dilationH
      << " dilationW: " << dilationW;

  int ndim = input.ndims();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  PARROTS_CHECKARGS(ndim == 3 || ndim == 4)
      << "3D or 4D input tensor expected but got: " << ndim;

  size_t nInputPlane = weight.dim(1) * group;
  size_t inputHeight = input.dim(dimh);
  size_t inputWidth = input.dim(dimw);
  size_t nOutputPlane = weight.dim(0);
  size_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  size_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  PARROTS_CHECKARGS(nInputPlane % deformable_group == 0)
      << "input channels must divide deformable group size";

  PARROTS_CHECKARGS(outputWidth >= 1 || outputHeight >= 1)
      << "Given input size: (" << nInputPlane << " x " << inputHeight << " x "
      << inputWidth << "). Calculated output size: (" << nOutputPlane << " x "
      << outputHeight << " x " << outputWidth << "). Output size is too small";

  PARROTS_CHECKARGS(input.dim(1) == nInputPlane)
      << "invalid number of input planes, expected: " << nInputPlane
      << ", but got: " << input.dim(1);

  PARROTS_CHECKARGS(inputHeight >= kH && inputWidth >= kW)
      << "input image is smaller than kernel";

  PARROTS_CHECKARGS(offset.dim(2) == outputHeight &&
                    offset.dim(3) == outputWidth)
      << "invalid spatial dim of offset, expected height: " << outputHeight
      << " width: " << outputWidth << ", but got height: " << offset.dim(2)
      << " width: " << offset.dim(3);

  PARROTS_CHECKARGS(offset.dim(1) == deformable_group * 2 * kH * kW)
      << "invalid number of channels of offset";

  if (gradOutput != NULL) {
    PARROTS_CHECKARGS(gradOutput->dim(dimf) == nOutputPlane)
        << "invalid number of gradOutput planes, expected: " << nOutputPlane
        << ", but got: " << gradOutput->dim(dimf);

    PARROTS_CHECKARGS(gradOutput->dim(dimh) == outputHeight &&
                      gradOutput->dim(dimw) == outputWidth)
        << "invalid dim of gradOutput, expected height: " << outputHeight
        << " width: " << outputWidth
        << " , but got height: " << gradOutput->dim(dimh)
        << " width: " << gradOutput->dim(dimw);
  }
}

void DeformConvForwardCUDAKernelLauncher(
    DArrayLite input, DArrayLite weight, DArrayLite offset, DArrayLite output,
    DArrayLite columns, DArrayLite ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step, CudaContext& ctx,
    cudaStream_t stream) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  deform_conv_shape_check(input, offset, NULL, weight, kH, kW, dH, dW, padH,
                          padW, dilationH, dilationW, group, deformable_group);

  int batch = 1;
  if (input.ndims() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
    offset = offset.view({1, offset.dim(0), offset.dim(1), offset.dim(2)});
  }

  // todo: assert batchsize dividable by im2col_step

  size_t batchSize = input.dim(0);
  size_t nInputPlane = input.dim(1);
  size_t inputHeight = input.dim(2);
  size_t inputWidth = input.dim(3);

  size_t nOutputPlane = weight.dim(0);

  size_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  size_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  PARROTS_CHECKARGS(offset.dim(0) == batchSize)
      << "invalid batch size of offset";

  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane,
                        outputHeight, outputWidth});

  columns = ctx.createDArrayLite(
      input.elemType(), DArrayShape(nInputPlane * kW * kH,
                                    im2col_step * outputHeight * outputWidth));
  columns.setZeros(ctx.getStream());

  if (ones.ndims() != 2 ||
      ones.dim(0) * ones.dim(1) < outputHeight * outputWidth) {
    ones = ctx.createDArrayLite(input.elemType(),
                                DArrayShape(outputHeight, outputWidth));
    fill(ctx, ones, *toScalar(1));
  }

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  auto output_buffer = ctx.createDArrayLite(
      input.elemType(), DArrayShape(batchSize / im2col_step, nOutputPlane,
                                    im2col_step * outputHeight, outputWidth));
  output_buffer.setZeros(ctx.getStream());
  output_buffer = output_buffer.view(
      {output_buffer.dim(0), group, output_buffer.dim(1) / group,
       output_buffer.dim(2) * output_buffer.dim(3)});

  for (size_t elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns,
                      stream);

    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
    weight = weight.view(
        {group, nOutputPlane / group, nInputPlane / group * kH * kW});

    for (size_t g = 0; g < group; g++) {
      auto output_g = output_buffer[elt][g];
      auto weight_g = weight[g];
      auto columns_g = columns[g];
      gemm(ctx, 1, false, weight_g, false, columns_g, 1, output_g);
    }
    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    weight = weight.view({nOutputPlane, nInputPlane, kH, kW});
  }

  output_buffer = output_buffer.view(
      {output_buffer.dim(0), output_buffer.dim(1) * output_buffer.dim(2),
       output_buffer.dim(3)});

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                      im2col_step, outputHeight, outputWidth});
  output_buffer = transpose(ctx, output_buffer, 1, 2);
  if (!output_buffer.isContiguous()) {
    output_buffer = ctx.cloneDArrayLite(output_buffer);
  }
  copy(ctx, output, output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.dim(1), offset.dim(2), offset.dim(3)});
  }
}

void DeformConvBackwardInputCUDAKernelLauncher(
    DArrayLite input, DArrayLite offset, DArrayLite gradOutput,
    DArrayLite gradInput, DArrayLite gradOffset, DArrayLite weight,
    DArrayLite columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int deformable_group,
    int im2col_step, CudaContext& ctx, cudaStream_t stream) {
  deform_conv_shape_check(input, offset, &gradOutput, weight, kH, kW, dH, dW,
                          padH, padW, dilationH, dilationW, group,
                          deformable_group);

  int batch = 1;

  if (input.ndims() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
    offset = offset.view({1, offset.dim(0), offset.dim(1), offset.dim(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.dim(0), gradOutput.dim(1), gradOutput.dim(2)});
  }

  size_t batchSize = input.dim(0);
  size_t nInputPlane = input.dim(1);
  size_t inputHeight = input.dim(2);
  size_t inputWidth = input.dim(3);

  size_t nOutputPlane = weight.dim(0);

  size_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  size_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  PARROTS_CHECKARGS(offset.dim(0) == batchSize)
      << "invalid batch size of offset";
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = ctx.createDArrayLite(
      input.elemType(), DArrayShape(nInputPlane * kW * kH,
                                    im2col_step * outputHeight * outputWidth));
  columns.setZeros(ctx.getStream());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput = transpose(ctx, gradOutput, 1, 2);
  if (!gradOutput.isContiguous()) {
    gradOutput = ctx.cloneDArrayLite(gradOutput);
  }

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

  for (size_t elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
    weight = weight.view({group, weight.dim(0) / group,
                          weight.dim(1) * weight.dim(2) * weight.dim(3)});
    gradOutput = gradOutput.view(
        {gradOutput.dim(0), group, gradOutput.dim(1) / group,
         gradOutput.dim(2) * gradOutput.dim(3) * gradOutput.dim(4)});

    for (size_t g = 0; g < group; g++) {
      auto columns_g = columns[g];
      gemm(ctx, 1, true, weight[g], false, gradOutput[elt][g], 0, columns_g);
    }

    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    gradOutput = gradOutput.view({gradOutput.dim(0),
                                  gradOutput.dim(1) * gradOutput.dim(2),
                                  im2col_step, outputHeight, outputWidth});
    weight = weight.view({nOutputPlane, nInputPlane, kH, kW});

    deformable_col2im_coord(columns, input[elt], offset[elt], nInputPlane,
                            inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW, im2col_step, deformable_group,
                            gradOffset[elt], stream);

    deformable_col2im(columns, offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, gradInput[elt],
                      stream);
  }

  gradOutput = transpose(ctx, gradOutput, 1, 2);
  if (!gradOutput.isContiguous()) {
    gradOutput = ctx.cloneDArrayLite(gradOutput);
  }
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
    offset = offset.view({offset.dim(1), offset.dim(2), offset.dim(3)});
    gradOffset = gradOffset.view({offset.dim(1), offset.dim(2), offset.dim(3)});
  }
}

void DeformConvBackwardParametersCUDAKernelLauncher(
    DArrayLite input, DArrayLite offset, DArrayLite gradOutput,
    DArrayLite gradWeight, DArrayLite columns, DArrayLite ones, int kW, int kH,
    int dW, int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, float scale, int im2col_step, CudaContext& ctx,
    cudaStream_t stream) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input

  deform_conv_shape_check(input, offset, &gradOutput, gradWeight, kH, kW, dH,
                          dW, padH, padW, dilationH, dilationW, group,
                          deformable_group);

  int batch = 1;

  if (input.ndims() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.dim(0), input.dim(1), input.dim(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.dim(0), gradOutput.dim(1), gradOutput.dim(2)});
  }

  size_t batchSize = input.dim(0);
  size_t nInputPlane = input.dim(1);
  size_t inputHeight = input.dim(2);
  size_t inputWidth = input.dim(3);

  size_t nOutputPlane = gradWeight.dim(0);

  size_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  size_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  PARROTS_CHECKARGS(offset.dim(0) == batchSize)
      << "invalid batch size of offset";

  columns = ctx.createDArrayLite(
      input.elemType(), DArrayShape(nInputPlane * kW * kH,
                                    im2col_step * outputHeight * outputWidth));
  columns.setZeros(ctx.getStream());

  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput = transpose(ctx, gradOutput, 1, 2);
  if (!gradOutput.isContiguous()) {
    gradOutput = ctx.cloneDArrayLite(gradOutput);
  }

  auto gradOutputBuffer = ctx.cloneDArrayLite(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane,
                             im2col_step * outputHeight, outputWidth});

  gradOutput = transpose(ctx, gradOutput, 1, 2);
  if (!gradOutput.isContiguous()) {
    gradOutput = ctx.cloneDArrayLite(gradOutput);
  }
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (size_t elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns,
                      stream);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.dim(0), group, gradOutputBuffer.dim(1) / group,
         gradOutputBuffer.dim(2) * gradOutputBuffer.dim(3)});
    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
    gradWeight = gradWeight.view(
        {group, gradWeight.dim(0) / group,
         gradWeight.dim(1) * gradWeight.dim(2) * gradWeight.dim(3)});

    for (int g = 0; g < group; g++) {
      auto gradWeight_g = gradWeight[g];
      gemm(ctx, scale, false, gradOutputBuffer[elt][g], true, columns[g], 1,
           gradWeight_g);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.dim(0),
         gradOutputBuffer.dim(1) * gradOutputBuffer.dim(2),
         im2col_step * outputHeight, outputWidth});
    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    gradWeight = gradWeight.view(
        {gradWeight.dim(0) * gradWeight.dim(1), nInputPlane / group, kH, kW});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }
}
