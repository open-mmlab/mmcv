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

void bbox_overlaps_mlu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                       const int32_t mode, const bool aligned,
                       const int32_t offset) {
  // check dtype
  TORCH_CHECK(
      bboxes1.scalar_type() == at::kFloat || bboxes1.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      bboxes1.scalar_type(), ".");
  TORCH_CHECK(bboxes1.scalar_type() == bboxes2.scalar_type(),
              "bboxes1's dtype should be the same with bboxes2's dtype.");

  // params check
  TORCH_CHECK(bboxes1.dim() == 2, "bboxes1 should be a 2d tensor, got ",
              bboxes1.dim(), "D");
  TORCH_CHECK(bboxes2.dim() == 2, "bboxes2 should be a 2d tensor, got ",
              bboxes2.dim(), "D");

  auto rows = bboxes1.size(0);
  auto cols = bboxes2.size(0);
  auto batch_num_all = rows;

  if (rows * cols == 0) {
    // return if zero element
    return;
  }

  INITIAL_MLU_PARAM_WITH_TENSOR(bboxes1);
  INITIAL_MLU_PARAM_WITH_TENSOR(bboxes2);
  INITIAL_MLU_PARAM_WITH_TENSOR(ious);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  TORCH_MLUOP_CHECK(mluOpBboxOverlaps(
      handle, mode, aligned, offset, bboxes1_desc.desc(), bboxes1_ptr,
      bboxes2_desc.desc(), bboxes2_ptr, ious_desc.desc(), ious_ptr));
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

REGISTER_DEVICE_IMPL(bbox_overlaps_impl, MLU, bbox_overlaps_mlu);
