/*************************************************************************
 * Copyright (C) 2021 by Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include "pytorch_mlu_helper.hpp"
#include "pytorch_device_registry.hpp"

void KernelNms(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
               const cnrtDataType_t data_type_input, const void *boxes_ptr,
               const void *scores_ptr, const int input_num_boxes,
               const int input_stride, const int max_output_boxes,
               const float iou_threshold, const float offset,
               void *workspace_ptr, void *output_size_ptr, void *output_ptr);

Tensor NMSMLUKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                            int offset) {
  // dimension parameters check
  TORCH_CHECK(boxes.dim() == 2, "boxes should be a 2d tensor, got ",
              boxes.dim(), "D");
  TORCH_CHECK(boxes.size(1) == 4,
              "boxes should have 4 elements in dimension 1, got ",
              boxes.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ",
              scores.dim(), "D");

  // data type check
  TORCH_CHECK(boxes.scalar_type() == scores.scalar_type(),
              "boxes should have the same type as scores");
  TORCH_CHECK(
      boxes.scalar_type() == at::kFloat || boxes.scalar_type() == at::kHalf,
      "data type of boxes should be Float or Half, got ", boxes.scalar_type());

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  int input_num_boxes = boxes.size(0);
  int input_stride = boxes.size(1);
  int max_output_boxes = boxes.size(0);
  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_x = core_dim;
  cnrtDim3_t k_dim = {dim_x, 1, 1};
  cnrtDataType_t data_type_input = torch_mlu::toCnrtDtype(boxes.dtype());

  auto output = at::empty({max_output_boxes}, boxes.options().dtype(at::kLong));
  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));

  // workspace
  size_t space_size = 0;
  if (boxes.scalar_type() == at::kHalf) {
    space_size = input_num_boxes * sizeof(int16_t);
  } else {
    space_size = input_num_boxes * sizeof(float);
  }
  auto workspace = at::empty(space_size, boxes.options().dtype(at::kByte));

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  auto boxes_impl = torch_mlu::getMluTensorImpl(boxes);
  auto boxes_ptr = boxes_impl->cnnlMalloc();
  auto scores_impl = torch_mlu::getMluTensorImpl(scores);
  auto scores_ptr = scores_impl->cnnlMalloc();
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  auto output_size_impl = torch_mlu::getMluTensorImpl(output_size);
  auto output_size_ptr = output_size_impl->cnnlMalloc();

  switch (k_type) {
    default: {
      TORCH_CHECK(false, "[nms_mlu]:Failed to choose kernel to launch");
    }
    case CNRT_FUNC_TYPE_BLOCK:
    case CNRT_FUNC_TYPE_UNION1: {
      CNLOG(INFO) << "Launch Kernel MLUUnion1 or Block NMS<<<Union"
                  << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y
                  << ", " << k_dim.z << ">>>";
      KernelNms(k_dim, k_type, queue, data_type_input, boxes_ptr, scores_ptr,
                input_num_boxes, input_stride, max_output_boxes, iou_threshold,
                offset, workspace_ptr, output_size_ptr, output_ptr);
    }; break;
  }

  int output_num = *static_cast<int *>(output_size.cpu().data_ptr());
  return output.slice(0, 0, output_num);
}

Tensor nms_mlu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSMLUKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, MLU, nms_mlu);
