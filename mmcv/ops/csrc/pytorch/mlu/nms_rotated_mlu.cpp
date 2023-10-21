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

Tensor nms_rotated_mlu(Tensor boxes, Tensor scores, float iou_threshold) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  int boxes_num = boxes.size(0);
  auto boxes_ = torch_mlu::cnnl::ops::cnnl_contiguous(boxes);
  auto scores_ = torch_mlu::cnnl::ops::cnnl_contiguous(scores);
  auto output = at::empty({boxes_num}, boxes.options().dtype(at::kInt));
  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));

  MluOpTensorDescriptor boxes_desc, scores_desc, output_desc;
  boxes_desc.set(boxes_);
  scores_desc.set(scores_);
  output_desc.set(output);

  // workspace
  size_t workspace_size = 0;
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpGetNmsRotatedWorkspaceSize(handle, boxes_desc.desc(),
                                                    &workspace_size));
  auto workspace = at::empty(workspace_size, boxes.options().dtype(at::kByte));

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

  TORCH_MLUOP_CHECK(mluOpNmsRotated(
      handle, iou_threshold, boxes_desc.desc(), boxes_ptr, scores_desc.desc(),
      scores_ptr, workspace_ptr, workspace_size, output_desc.desc(), output_ptr,
      (int *)output_size_ptr));
  int output_num = *static_cast<int *>(output_size.cpu().data_ptr());
  auto ret = output.to(boxes.options().dtype(at::kLong));
  return ret.slice(0, 0, output_num);
}
