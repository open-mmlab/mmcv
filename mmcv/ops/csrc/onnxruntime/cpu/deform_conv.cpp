// Copyright (c) OpenMMLab. All rights reserved
#include "deform_conv.h"

#include "../ort_mmcv_utils.h"
#include <torch/torch.h>
#include <vector>

at::Tensor ort_to_tensor(Ort::CustomOpApi &ort, const OrtValue *value) {
  at::Tensor tensor =
      at::from_blob((void *)ort.GetTensorData<float>(value),
                    ort.GetTensorShape(ort.GetTensorTypeAndShape(value)));
  return tensor;
}

template <typename T>
T deformable_im2col_bilinear_cpu(const T *input, const int64_t data_width,
                                 const int64_t height, const int64_t width, T h,
                                 T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int64_t h_low = floor(h);
  int64_t w_low = floor(w);
  int64_t h_high = h_low + 1;
  int64_t w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
void deformable_im2col_cpu_kernel(
    const int64_t n, const T *data_im, const T *data_offset,
    const int64_t height, const int64_t width, const int64_t kernel_h,
    const int64_t kernel_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t dilation_h,
    const int64_t dilation_w, const int64_t channel_per_deformable_group,
    const int64_t batch_size, const int64_t num_channels,
    const int64_t deformable_group, const int64_t height_col,
    const int64_t width_col, T *data_col) {
  for (int64_t index = 0; index < n; index++) {
    // index index of output matrix
    const int64_t w_col = index % width_col;
    const int64_t h_col = (index / width_col) % height_col;
    const int64_t b_col = (index / width_col / height_col) % batch_size;
    const int64_t c_im = (index / width_col / height_col) / batch_size;
    const int64_t c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int64_t deformable_group_index = c_im / channel_per_deformable_group;

    const int64_t h_in = h_col * stride_h - pad_h;
    const int64_t w_in = w_col * stride_w - pad_w;
    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;

    for (int64_t i = 0; i < kernel_h; ++i) {
      for (int64_t j = 0; j < kernel_w; ++j) {
        const int64_t data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int64_t data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          val = deformable_im2col_bilinear_cpu(data_im_ptr, width, height,
                                               width, h_im, w_im);
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

void deformable_im2col(at::Tensor data_im, at::Tensor data_offset,
                       const int64_t channels, const int64_t height,
                       const int64_t width, const int64_t ksize_h,
                       const int64_t ksize_w, const int64_t pad_h,
                       const int64_t pad_w, const int64_t stride_h,
                       const int64_t stride_w, const int64_t dilation_h,
                       const int64_t dilation_w, const int64_t parallel_imgs,
                       const int64_t deformable_group, at::Tensor data_col) {
  int64_t height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int64_t width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int64_t num_kernels = channels * height_col * width_col * parallel_imgs;
  int64_t channel_per_deformable_group = channels / deformable_group;
  deformable_im2col_cpu_kernel<float>(
      num_kernels, data_im.data_ptr<float>(), data_offset.data_ptr<float>(),
      height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs,
      channels, deformable_group, height_col, width_col,
      data_col.data_ptr<float>());
}

MMCVDeformConvKernel::MMCVDeformConvKernel(OrtApi api,
                                           const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  std::vector<int64_t> stride =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "stride");
  stride_height_ = stride[0];
  stride_width_ = stride[1];
  std::vector<int64_t> padding =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "padding");
  padding_height_ = padding[0];
  padding_width_ = padding[1];
  std::vector<int64_t> dilation =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "dilation");
  dilation_height_ = dilation[0];
  dilation_width_ = dilation[1];
  deformable_group_ =
      ort_.KernelInfoGetAttribute<int64_t>(info, "deform_groups");
  group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "groups");

  im2col_step_ = ort_.KernelInfoGetAttribute<int64_t>(info, "im2col_step");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void MMCVDeformConvKernel::Compute(OrtKernelContext *context) {

  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  at::Tensor input_data = ort_to_tensor(ort_, input);

  const OrtValue *offset = ort_.KernelContext_GetInput(context, 1);
  at::Tensor offset_data = ort_to_tensor(ort_, offset);

  const OrtValue *filter = ort_.KernelContext_GetInput(context, 2);
  at::Tensor filter_data = ort_to_tensor(ort_, filter);

  OrtTensorDimensions input_dims(ort_, input);
  OrtTensorDimensions filter_dims(ort_, filter);

  const int64_t batch_size = input_dims[0];
  const int64_t in_channels = input_dims[1];
  const int64_t in_height = input_dims[2];
  const int64_t in_width = input_dims[3];
  const int64_t out_channels = filter_dims[0];
  const int64_t kernel_height = filter_dims[2];
  const int64_t kernel_width = filter_dims[3];

  const int64_t stride_height = stride_height_;
  const int64_t stride_width = stride_width_;
  const int64_t padding_height = padding_height_;
  const int64_t padding_width = padding_width_;
  const int64_t dilation_height = dilation_height_;
  const int64_t dilation_width = dilation_width_;
  const int64_t deformable_group = deformable_group_;
  const int64_t im2col_step = std::min(im2col_step_, batch_size);
  const int64_t group = group_;

  // get output memory
  const int64_t out_height = floor((in_height + 2 * padding_height -
                                    dilation_height * (kernel_height - 1) - 1) /
                                       stride_height +
                                   1);
  const int64_t out_width = floor(
      (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) /
          stride_width +
      1);

  std::vector<int64_t> output_dims = {batch_size, out_channels, out_height,
                                      out_width};

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, output_dims.data(), output_dims.size());
  float *out_ptr = ort_.GetTensorMutableData<float>(output);

  int batch = 1;
  if (input_dims.size() == 3) {
    batch = 0;
    input_data.unsqueeze_(0);
    offset_data.unsqueeze_(0);
  }

  at::Tensor output_data = at::zeros({batch_size / im2col_step, im2col_step,
                                      out_channels, out_height, out_width},
                                     input_data.options());
  at::Tensor columns = at::zeros({in_channels * kernel_width * kernel_height,
                                  im2col_step * out_height * out_width},
                                 input_data.options());

  at::Tensor output_buffer = at::zeros({batch_size / im2col_step, out_channels,
                                        im2col_step * out_height, out_width},
                                       output_data.options());

  input_data = input_data.view({batch_size / im2col_step, im2col_step,
                                in_channels, in_height, in_width});
  offset_data =
      offset_data.view({batch_size / im2col_step, im2col_step,
                        deformable_group * 2 * kernel_height * kernel_width,
                        out_height, out_width});

  output_buffer = output_buffer.view(
      {output_buffer.size(0), group, output_buffer.size(1) / group,
       output_buffer.size(2), output_buffer.size(3)});

  for (int64_t elt = 0; elt < batch_size / im2col_step; elt++) {
    deformable_im2col(input_data[elt], offset_data[elt], in_channels, in_height,
                      in_width, kernel_height, kernel_width, padding_height,
                      padding_width, stride_height, stride_width,
                      dilation_height, dilation_width, im2col_step,
                      deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    filter_data = filter_data.view({group, filter_data.size(0) / group,
                                    filter_data.size(1), filter_data.size(2),
                                    filter_data.size(3)});

    for (int64_t g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(filter_data[g].flatten(1), columns[g])
                                  .view_as(output_buffer[elt][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    filter_data = filter_data.view({filter_data.size(0) * filter_data.size(1),
                                    filter_data.size(2), filter_data.size(3),
                                    filter_data.size(4)});
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3), output_buffer.size(4)});

  output_buffer = output_buffer.view({batch_size / im2col_step, out_channels,
                                      im2col_step, out_height, out_width});
  output_buffer.transpose_(1, 2);
  output_data.copy_(output_buffer);
  output_data =
      output_data.view({batch_size, out_channels, out_height, out_width});

  if (batch == 0)
    output_data = output_data.view({out_channels, out_height, out_width});

  std::memcpy(out_ptr, output_data.data_ptr<float>(),
              sizeof(float) * batch_size * out_channels * out_height *
                  out_width);
}
