/*************************************************************************
 * Copyright (C) 2023 Cambricon.
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

Tensor diff_iou_rotated_sort_vertices_forward_mlu(Tensor vertices, Tensor mask,
                                                  Tensor num_valid) {
  // params check
  TORCH_CHECK(vertices.scalar_type() == at::kFloat,
              "vertices type should be Float, got ", vertices.scalar_type());
  TORCH_CHECK(mask.scalar_type() == at::kBool, "mask should be Bool, got ",
              mask.scalar_type());
  TORCH_CHECK(num_valid.scalar_type() == at::kInt,
              "num_valid type should be Int32, got ", num_valid.scalar_type());
  TORCH_CHECK(vertices.size(2) == 24, "vertices.dim(2) should be 24, got ",
              vertices.size(2));
  TORCH_CHECK(mask.size(2) == 24, "mask.dim(2) should be 24, got ",
              mask.size(2));

  // zero-element check
  if (vertices.numel() == 0) {
    return at::empty({0}, num_valid.options().dtype(at::kInt));
  }

  auto idx = at::empty({vertices.size(0), vertices.size(1), 9},
                       num_valid.options().dtype(at::kInt));

  INITIAL_MLU_PARAM_WITH_TENSOR(vertices);
  INITIAL_MLU_PARAM_WITH_TENSOR(mask);
  INITIAL_MLU_PARAM_WITH_TENSOR(num_valid);
  INITIAL_MLU_PARAM_WITH_TENSOR(idx);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpDiffIouRotatedSortVerticesForward(
      handle, vertices_desc.desc(), vertices_ptr, mask_desc.desc(), mask_ptr,
      num_valid_desc.desc(), num_valid_ptr, idx_desc.desc(), idx_ptr));
  return idx;
}

Tensor diff_iou_rotated_sort_vertices_forward_impl(Tensor vertices, Tensor mask,
                                                   Tensor num_valid);

REGISTER_DEVICE_IMPL(diff_iou_rotated_sort_vertices_forward_impl, MLU,
                     diff_iou_rotated_sort_vertices_forward_mlu);
