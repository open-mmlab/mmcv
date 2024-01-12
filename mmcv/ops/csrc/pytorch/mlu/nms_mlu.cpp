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

Tensor NMSMLUKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                            int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  int max_output_boxes = boxes.size(0);

  // transpose boxes (n, 4) to (4, n) for better performance
  auto boxes_ = torch_mlu::cnnl::ops::cnnl_contiguous(boxes);
  auto scores_ = torch_mlu::cnnl::ops::cnnl_contiguous(scores);
  auto output = at::empty({max_output_boxes}, boxes.options().dtype(at::kInt));
  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));

  MluOpTensorDescriptor boxes_desc, scores_desc, output_desc;
  boxes_desc.set(boxes_);
  scores_desc.set(scores_);
  output_desc.set(output);

  // workspace
  size_t workspace_size = 0;
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpGetNmsWorkspaceSize(
      handle, boxes_desc.desc(), scores_desc.desc(), &workspace_size));
  auto workspace = at::empty(workspace_size, boxes.options().dtype(at::kByte));

  // get compute queue
  auto boxes_impl = torch_mlu::getMluTensorImpl(boxes_);
  auto boxes_ptr = boxes_impl->cnnlMalloc();
  auto scores_impl = torch_mlu::getMluTensorImpl(scores_);
  auto scores_ptr = scores_impl->cnnlMalloc();
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  auto output_size_impl = torch_mlu::getMluTensorImpl(output_size);
  auto output_size_ptr = output_size_impl->cnnlMalloc();

  // nms desc
  mluOpNmsDescriptor_t nms_desc;
  const mluOpNmsBoxPointMode_t box_mode = (mluOpNmsBoxPointMode_t)0;
  const mluOpNmsOutputMode_t output_mode = (mluOpNmsOutputMode_t)0;
  const mluOpNmsAlgo_t algo = (mluOpNmsAlgo_t)0;
  const mluOpNmsMethodMode_t method_mode = (mluOpNmsMethodMode_t)0;
  const float soft_nms_sigma = 0.0;
  const float confidence_threshold = 0.0;
  const int input_layout = 0;
  const bool pad_to_max_output_size = false;
  const int max_output_size = max_output_boxes;

  TORCH_MLUOP_CHECK(mluOpCreateNmsDescriptor(&nms_desc));
  TORCH_MLUOP_CHECK(mluOpSetNmsDescriptor(
      nms_desc, box_mode, output_mode, algo, method_mode, iou_threshold,
      soft_nms_sigma, max_output_size, confidence_threshold, (float)offset,
      input_layout, pad_to_max_output_size));

  TORCH_MLUOP_CHECK(mluOpNms(handle, nms_desc, boxes_desc.desc(), boxes_ptr,
                             scores_desc.desc(), scores_ptr, workspace_ptr,
                             workspace_size, output_desc.desc(), output_ptr,
                             output_size_ptr));
  TORCH_MLUOP_CHECK(mluOpDestroyNmsDescriptor(nms_desc));
  int output_num = *static_cast<int *>(output_size.cpu().data_ptr());
  auto ret = output.to(boxes.options().dtype(at::kLong));
  return ret.slice(0, 0, output_num);
}

Tensor nms_mlu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSMLUKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, MLU, nms_mlu);
