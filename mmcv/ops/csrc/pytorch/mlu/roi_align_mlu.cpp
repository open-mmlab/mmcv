/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"

void ROIAlignForwardMLUKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax_y, Tensor argmax_x,
                                      int aligned_height, int aligned_width,
                                      float spatial_scale, int sampling_ratio,
                                      int pool_mode, bool aligned) {
  // params check
  TORCH_CHECK(pool_mode == 1, "pool_mode only supports 'avg' currently");
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_tensor =
      torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  auto output_contiguous =
      at::empty({num_rois, channels, aligned_height, aligned_width},
                input.options(), memory_format);
  // get tensor impl
  auto self_impl = torch_mlu::getMluTensorImpl(input_tensor);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);

  MluOpTensorDescriptor input_desc, rois_desc, argmax_y_desc, argmax_x_desc,
      output_desc;
  input_desc.set_with_layout(input_tensor, MLUOP_LAYOUT_NHWC);
  rois_desc.set_with_layout(rois, MLUOP_LAYOUT_ARRAY);
  output_desc.set_with_layout(output_contiguous, MLUOP_LAYOUT_NHWC);

  // get the mlu ptr
  auto self_ptr = self_impl->cnnlMalloc();
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  mluOpRoiAlignForwardDescriptor_t roialign_desc;
  TORCH_MLUOP_CHECK(mluOpCreateRoiAlignForwardDescriptor(&roialign_desc));
  TORCH_MLUOP_CHECK(mluOpSetRoiAlignForwardDescriptor_v2(
      roialign_desc, aligned_height, aligned_width, sampling_ratio,
      spatial_scale, pool_mode, aligned));

  auto handle = mluOpGetCurrentHandle();
  if (pool_mode == 0) {
    auto argmax_y_contiguous =
        torch_mlu::cnnl::ops::cnnl_contiguous(argmax_y, memory_format);
    auto argmax_x_contiguous =
        torch_mlu::cnnl::ops::cnnl_contiguous(argmax_x, memory_format);
    auto argmax_x_impl = torch_mlu::getMluTensorImpl(argmax_x_contiguous);
    auto argmax_y_impl = torch_mlu::getMluTensorImpl(argmax_y_contiguous);
    auto argmax_x_ptr = argmax_x_impl->cnnlMalloc();
    auto argmax_y_ptr = argmax_y_impl->cnnlMalloc();
    argmax_y_desc.set_with_layout(argmax_x_contiguous, MLUOP_LAYOUT_NHWC);
    argmax_x_desc.set_with_layout(argmax_x_contiguous, MLUOP_LAYOUT_NHWC);
    TORCH_MLUOP_CHECK(mluOpRoiAlignForward_v2(
        handle, roialign_desc, input_desc.desc(), self_ptr, rois_desc.desc(),
        rois_ptr, output_desc.desc(), output_ptr, argmax_x_desc.desc(),
        argmax_x_ptr, argmax_y_desc.desc(), argmax_y_ptr));
    argmax_x.copy_(argmax_x_contiguous);
    argmax_y.copy_(argmax_y_contiguous);
  } else {
    TORCH_MLUOP_CHECK(mluOpRoiAlignForward_v2(
        handle, roialign_desc, input_desc.desc(), self_ptr, rois_desc.desc(),
        rois_ptr, output_desc.desc(), output_ptr, NULL, NULL, NULL, NULL));
  }
  TORCH_MLUOP_CHECK(mluOpDestroyRoiAlignForwardDescriptor(roialign_desc));
  output.copy_(output_contiguous);
}

void ROIAlignBackwardMLUKernelLauncher(Tensor grad, Tensor rois,
                                       Tensor argmax_y, Tensor argmax_x,
                                       Tensor grad_input, int aligned_height,
                                       int aligned_width, float spatial_scale,
                                       int sampling_ratio, int pool_mode,
                                       bool aligned) {
  // params check
  TORCH_CHECK(pool_mode == 1, "pool_mode only supports 'avg' currently");
  int batch_size = grad_input.size(0);
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad.dim());
  auto grad_ = torch_mlu::cnnl::ops::cnnl_contiguous(grad, memory_format);
  auto grad_input_ = at::empty({batch_size, channels, height, width},
                               grad.options(), memory_format)
                         .zero_();

  int boxes_num = rois.size(0);
  int hi = grad.size(2);
  int wi = grad.size(3);
  int c = grad.size(1);

  int no = grad_input.size(0);
  int ho = grad_input.size(2);
  int wo = grad_input.size(3);

  // get tensor impl
  auto grad_impl = torch_mlu::getMluTensorImpl(grad_);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);

  // get the mlu ptr
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  MluOpTensorDescriptor grads_desc, rois_desc, argmax_y_desc, argmax_x_desc,
      grad_input_desc;
  grads_desc.set_with_layout(grad_, MLUOP_LAYOUT_NHWC);
  rois_desc.set_with_layout(rois, MLUOP_LAYOUT_ARRAY);
  grad_input_desc.set_with_layout(grad_input_, MLUOP_LAYOUT_NHWC);

  auto handle = mluOpGetCurrentHandle();
  if (pool_mode == 0) {
    auto argmax_y_contiguous =
        torch_mlu::cnnl::ops::cnnl_contiguous(argmax_y, memory_format);
    auto argmax_x_contiguous =
        torch_mlu::cnnl::ops::cnnl_contiguous(argmax_x, memory_format);
    auto argmax_x_impl = torch_mlu::getMluTensorImpl(argmax_x_contiguous);
    auto argmax_y_impl = torch_mlu::getMluTensorImpl(argmax_y_contiguous);
    auto argmax_x_ptr = argmax_x_impl->cnnlMalloc();
    auto argmax_y_ptr = argmax_y_impl->cnnlMalloc();
    argmax_y_desc.set_with_layout(argmax_x_contiguous, MLUOP_LAYOUT_NHWC);
    argmax_x_desc.set_with_layout(argmax_x_contiguous, MLUOP_LAYOUT_NHWC);
    TORCH_MLUOP_CHECK(mluOpRoiAlignBackward_v2(
        handle, grads_desc.desc(), grad_ptr, rois_desc.desc(), rois_ptr,
        argmax_y_desc.desc(), argmax_x_ptr, argmax_y_desc.desc(), argmax_y_ptr,
        spatial_scale, sampling_ratio, aligned, pool_mode,
        grad_input_desc.desc(), grad_input_ptr));
  } else {
    TORCH_MLUOP_CHECK(mluOpRoiAlignBackward_v2(
        handle, grads_desc.desc(), grad_ptr, rois_desc.desc(), rois_ptr, NULL,
        NULL, NULL, NULL, spatial_scale, sampling_ratio, aligned, pool_mode,
        grad_input_desc.desc(), grad_input_ptr));
  }
  grad_input.copy_(grad_input_);
}

void roi_align_forward_mlu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned) {
  ROIAlignForwardMLUKernelLauncher(input, rois, output, argmax_y, argmax_x,
                                   aligned_height, aligned_width, spatial_scale,
                                   sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_mlu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignBackwardMLUKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_DEVICE_IMPL(roi_align_forward_impl, MLU, roi_align_forward_mlu);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, MLU, roi_align_backward_mlu);
