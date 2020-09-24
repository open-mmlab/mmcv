#include "modulated_deform_conv_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void modulated_deformable_im2col_cuda(
    const DArrayLite data_im, const DArrayLite data_offset,
    const DArrayLite data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    DArrayLite data_col, cudaStream_t stream) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.elemType().prim(), ([&] {
        modulated_deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_im.ptr<scalar_t>(), data_offset.ptr<scalar_t>(),
            data_mask.ptr<scalar_t>(), height_im, width_im, kernel_h, kenerl_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size, channels,
            deformable_group, height_col, width_col, data_col.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_cuda(
    const DArrayLite data_col, const DArrayLite data_offset,
    const DArrayLite data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    DArrayLite grad_im, cudaStream_t stream) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.elemType().prim(), ([&] {
        modulated_deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_col.ptr<scalar_t>(), data_offset.ptr<scalar_t>(),
            data_mask.ptr<scalar_t>(), channels, height_im, width_im, kernel_h,
            kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size, deformable_group,
            height_col, width_col, grad_im.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_coord_cuda(
    const DArrayLite data_col, const DArrayLite data_im,
    const DArrayLite data_offset, const DArrayLite data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, DArrayLite grad_offset,
    DArrayLite grad_mask, cudaStream_t stream) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                          kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.elemType().prim(), ([&] {
        modulated_deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_col.ptr<scalar_t>(), data_im.ptr<scalar_t>(),
            data_offset.ptr<scalar_t>(), data_mask.ptr<scalar_t>(), channels,
            height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, 2 * kernel_h * kernel_w * deformable_group,
            deformable_group, height_col, width_col,
            grad_offset.ptr<scalar_t>(), grad_mask.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void ModulatedDeformConvForwardCUDAKernelLauncher(
    DArrayLite input, DArrayLite weight, DArrayLite bias, DArrayLite ones,
    DArrayLite offset, DArrayLite mask, DArrayLite output, DArrayLite columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias, CudaContext& ctx, cudaStream_t stream) {
  const int batch = input.dim(0);
  const int channels = input.dim(1);
  const int height = input.dim(2);
  const int width = input.dim(3);

  const int channels_out = weight.dim(0);
  const int channels_kernel = weight.dim(1);
  const int kernel_h_ = weight.dim(2);
  const int kernel_w_ = weight.dim(3);

  PARROTS_CHECKARGS(kernel_h_ == kernel_h && kernel_w_ == kernel_w)
      << "Input shape and kernel shape wont match: (" << kernel_h << " x "
      << kernel_w << " vs " << kernel_h_ << " x " << kernel_w_ << ").";

  PARROTS_CHECKARGS(channels == channels_kernel * group)
      << "Input shape and kernel channels wont match: (" << channels << " vs "
      << channels_kernel * group << ").";

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndims() != 2 || ones.dim(0) * ones.dim(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = ctx.createDArrayLite(input.elemType(),
                                DArrayShape(height_out, width_out));
    fill(ctx, ones, *toScalar(1));
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out});
  output.setZeros(ctx.getStream());

  // resize temporary columns
  columns = ctx.createDArrayLite(
      input.elemType(),
      DArrayShape(channels * kernel_h * kernel_w, 1 * height_out * width_out));
  columns.setZeros(ctx.getStream());

  output = output.view({output.dim(0), group, output.dim(1) / group,
                        output.dim(2), output.dim(3)});

  for (size_t b = 0; b < batch; b++) {
    modulated_deformable_im2col_cuda(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns, stream);

    // divide into group
    weight = weight.view({group, weight.dim(0) / group, weight.dim(1),
                          weight.dim(2), weight.dim(3)});
    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});

    for (size_t g = 0; g < group; g++) {
      auto output_g = output[b][g];
      gemm(ctx, 1, false,
           weight[g].view(
               {weight.dim(1), weight.dim(2) * weight.dim(3) * weight.dim(4)}),
           false, columns[g], 1, output_g);
    }

    weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2),
                          weight.dim(3), weight.dim(4)});
    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
  }

  output = output.view({output.dim(0), output.dim(1) * output.dim(2),
                        output.dim(3), output.dim(4)});

  if (with_bias) {
    bias = bias.view({1, bias.dim(0), 1, 1});
    add(ctx, output, bias, output);
  }
}

void ModulatedDeformConvBackwardCUDAKernelLauncher(
    DArrayLite input, DArrayLite weight, DArrayLite bias, DArrayLite ones,
    DArrayLite offset, DArrayLite mask, DArrayLite columns,
    DArrayLite grad_input, DArrayLite grad_weight, DArrayLite grad_bias,
    DArrayLite grad_offset, DArrayLite grad_mask, DArrayLite grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias, CudaContext& ctx, cudaStream_t stream) {
  const int batch = input.dim(0);
  const int channels = input.dim(1);
  const int height = input.dim(2);
  const int width = input.dim(3);

  const int channels_kernel = weight.dim(1);
  const int kernel_h_ = weight.dim(2);
  const int kernel_w_ = weight.dim(3);

  PARROTS_CHECKARGS(kernel_h_ == kernel_h && kernel_w_ == kernel_w)
      << "Input shape and kernel shape wont match: (" << kernel_h << " x "
      << kernel_w << " vs " << kernel_h_ << " x " << kernel_w_ << ").";

  PARROTS_CHECKARGS(channels == channels_kernel * group)
      << "Input shape and kernel channels wont match: (" << channels << " vs "
      << channels_kernel * group << ").";

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndims() != 2 || ones.dim(0) * ones.dim(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = ctx.createDArrayLite(input.elemType(),
                                DArrayShape(height_out, width_out));
    fill(ctx, ones, *toScalar(1));
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = ctx.createDArrayLite(
      input.elemType(),
      DArrayShape(channels * kernel_h * kernel_w, height_out * width_out));

  grad_output =
      grad_output.view({grad_output.dim(0), group, grad_output.dim(1) / group,
                        grad_output.dim(2), grad_output.dim(3)});

  for (size_t b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
    weight = weight.view({group, weight.dim(0) / group, weight.dim(1),
                          weight.dim(2), weight.dim(3)});

    for (size_t g = 0; g < group; g++) {
      auto columns_g = ctx.createDArrayLite(
          weight.elemType(), DArrayShape(columns.dim(1), columns.dim(2)));
      copy(ctx, columns_g, columns[g]);
      auto weight_g = weight[g].view(
          {weight.dim(1), weight.dim(2) * weight.dim(3) * weight.dim(4)});
      weight_g = transpose(ctx, weight_g, 0, 1);

      auto grad_output_bg = ctx.createDArrayLite(
          grad_output.elemType(),
          DArrayShape(grad_output.dim(2), grad_output.dim(3),
                      grad_output.dim(4)));
      copy(ctx, grad_output_bg, grad_output[b][g]);
      grad_output_bg =
          grad_output_bg.view({grad_output_bg.dim(0),
                               grad_output_bg.dim(1) * grad_output_bg.dim(2)});

      columns_g =
          parrots::op::addmm(ctx, columns[g], weight_g, grad_output_bg, 0, 1);
      auto columns_out = columns[g];
      copy(ctx, columns_out, columns_g);
    }

    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2),
                          weight.dim(3), weight.dim(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_cuda(
        columns, input[b], offset[b], mask[b], 1, channels, height, width,
        height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, deformable_group, grad_offset[b],
        grad_mask[b], stream);
    // gradient w.r.t. input data
    modulated_deformable_col2im_cuda(
        columns, offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, grad_input[b], stream);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_cuda(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns, stream);

    columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
    grad_weight =
        grad_weight.view({group, grad_weight.dim(0) / group, grad_weight.dim(1),
                          grad_weight.dim(2), grad_weight.dim(3)});
    if (with_bias) {
      grad_bias = grad_bias.view({group, grad_bias.dim(0) / group});
    }

    for (size_t g = 0; g < group; g++) {
      auto grad_weight_g = ctx.createDArrayLite(
          grad_weight.elemType(),
          DArrayShape(grad_weight.dim(1), grad_weight.dim(2),
                      grad_weight.dim(3), grad_weight.dim(4)));
      copy(ctx, grad_weight_g, grad_weight[g]);
      grad_weight_g = grad_weight_g.view(
          {grad_weight_g.dim(0),
           grad_weight_g.dim(1) * grad_weight_g.dim(2) * grad_weight_g.dim(3)});

      auto columns_g = columns[g];
      columns_g = transpose(ctx, columns_g, 0, 1);

      auto grad_output_bg = ctx.createDArrayLite(
          grad_output.elemType(),
          DArrayShape(grad_output.dim(2), grad_output.dim(3),
                      grad_output.dim(4)));
      copy(ctx, grad_output_bg, grad_output[b][g]);
      grad_output_bg =
          grad_output_bg.view({grad_output_bg.dim(0),
                               grad_output_bg.dim(1) * grad_output_bg.dim(2)});

      grad_weight_g = parrots::op::addmm(ctx, grad_weight_g, grad_output_bg,
                                         columns_g, 1, 1);
      auto grad_weight_out = grad_weight[g];
      copy(ctx, grad_weight_out, grad_weight_g);

      if (with_bias) {
        auto grad_bias_g = ctx.createDArrayLite(grad_bias.elemType(),
                                                DArrayShape(grad_bias.dim(1)));
        copy(ctx, grad_bias_g, grad_bias[g]);
        grad_bias_g = grad_bias_g.view({grad_bias_g.dim(0), 1});

        auto grad_output_bg = ctx.createDArrayLite(
            grad_output.elemType(),
            DArrayShape(grad_output.dim(2), grad_output.dim(3),
                        grad_output.dim(4)));
        copy(ctx, grad_output_bg, grad_output[b][g]);
        grad_output_bg = grad_output_bg.view(
            {grad_output_bg.dim(0),
             grad_output_bg.dim(1) * grad_output_bg.dim(2)});

        auto ones_g = ctx.createDArrayLite(
            ones.elemType(), DArrayShape(ones.dim(0), ones.dim(1)));
        copy(ctx, ones_g, ones);
        ones_g = ones_g.view({ones_g.dim(0) * ones_g.dim(1), 1});

        grad_bias_g =
            parrots::op::addmm(ctx, grad_bias_g, grad_output_bg, ones_g, 1, 1);

        auto grad_bias_out = grad_bias[g];
        copy(ctx, grad_bias_out, grad_bias_g);
      }
    }

    columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
    grad_weight = grad_weight.view({grad_weight.dim(0) * grad_weight.dim(1),
                                    grad_weight.dim(2), grad_weight.dim(3),
                                    grad_weight.dim(4)});
    if (with_bias)
      grad_bias =
          grad_bias.view(DArrayShape{grad_bias.dim(0) * grad_bias.dim(1)});
  }
  grad_output = grad_output.view({grad_output.dim(0) * grad_output.dim(1),
                                  grad_output.dim(2), grad_output.dim(3),
                                  grad_output.dim(4)});
}
