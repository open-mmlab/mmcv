/*************************************************************************
 * Copyright (C) 2022 by Cambricon.
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

void BoxIouRotatedMLUKernelLauncher(const Tensor boxes1, const Tensor boxes2,
                                    Tensor ious, const int mode_flag,
                                    const bool aligned) {
  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  auto boxes1_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      boxes1, boxes1.suggest_memory_format());
  auto boxes2_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      boxes2, boxes2.suggest_memory_format());
  auto ious_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(ious, ious.suggest_memory_format());

  MluOpTensorDescriptor boxes1_desc, boxes2_desc, ious_desc;
  boxes1_desc.set(boxes1_contiguous);
  boxes2_desc.set(boxes2_contiguous);
  ious_desc.set(ious_contiguous);

  auto boxes1_impl = torch_mlu::getMluTensorImpl(boxes1_contiguous);
  auto boxes2_impl = torch_mlu::getMluTensorImpl(boxes2_contiguous);
  auto ious_impl = torch_mlu::getMluTensorImpl(ious_contiguous);

  auto boxes1_ptr = boxes1_impl->cnnlMalloc();
  auto boxes2_ptr = boxes2_impl->cnnlMalloc();
  auto ious_ptr = ious_impl->cnnlMalloc();

  CNLOG(INFO) << "Call mluOpBoxIouRotated().";
  TORCH_MLUOP_CHECK(mluOpBoxIouRotated(
      handle, mode_flag, aligned, boxes1_desc.desc(), boxes1_ptr,
      boxes2_desc.desc(), boxes2_ptr, ious_desc.desc(), ious_ptr));
}

void box_iou_rotated_mlu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned) {
  BoxIouRotatedMLUKernelLauncher(boxes1, boxes2, ious, mode_flag, aligned);
}

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);

REGISTER_DEVICE_IMPL(box_iou_rotated_impl, MLU, box_iou_rotated_mlu);
