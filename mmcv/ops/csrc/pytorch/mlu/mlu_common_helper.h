/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#pragma once
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include "aten.h"
#include "mlu_op.h"
#include "pytorch_device_registry.hpp"

#define MLUOP_MAJOR 0
#define MLUOP_MINOR 8
#define MLUOP_PATCHLEVEL 1

/*************************************************************************
 * This MACRO contains operations of simple tensor to mlu-tensor.
 * _contiguous, _desc, _impl, _ptr will be automatically generated in
 * this MACRO.
 *************************************************************************/
#define INITIAL_MLU_PARAM_WITH_TENSOR(NAME)                         \
  auto NAME##_contigous = torch_mlu::cnnl::ops::cnnl_contiguous(    \
      NAME, NAME.suggest_memory_format());                          \
  MluOpTensorDescriptor NAME##_desc;                                \
  NAME##_desc.set(NAME##_contigous);                                \
  auto NAME##_impl = torch_mlu::getMluTensorImpl(NAME##_contigous); \
  auto NAME##_ptr = NAME##_impl->cnnlMalloc();

#ifndef TORCH_MLUOP_CHECK
#define TORCH_MLUOP_CHECK(EXPR)                                          \
  do {                                                                   \
    mluOpStatus_t status = EXPR;                                         \
    if (status != MLUOP_STATUS_SUCCESS) {                                \
      CNLOG(ERROR) << "";                                                \
      TORCH_CHECK(false, "MLUOPS error: ", mluOpGetErrorString(status)); \
    }                                                                    \
  } while (0);
#endif

enum class reduce_t { SUM = 0, MEAN = 1, MAX = 2 };

inline std::string to_string(reduce_t reduce_type) {
  if (reduce_type == reduce_t::MAX) {
    return "max";
  } else if (reduce_type == reduce_t::MEAN) {
    return "mean";
  } else if (reduce_type == reduce_t::SUM) {
    return "sum";
  } else {
    return "unknown reduce type";
  }
}

mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type);
mluOpTensorLayout_t getMluOpSuggestLayout(const at::Tensor& input);
mluOpReduceMode_t getMluOpReduceMode(const reduce_t reduce_type);

class MluOpTensorDescriptor {
 public:
  MluOpTensorDescriptor() {
    TORCH_MLUOP_CHECK(mluOpCreateTensorDescriptor(&desc_));
  };
  ~MluOpTensorDescriptor() {
    TORCH_MLUOP_CHECK(mluOpDestroyTensorDescriptor(desc_));
  }

  void set(at::Tensor);
  void set_with_layout(at::Tensor, mluOpTensorLayout_t layout);
  mluOpTensorDescriptor_t desc() { return desc_; }

 private:
  mluOpTensorDescriptor_t desc_;
  void set_desc(const at::Tensor&, mluOpTensorLayout_t, mluOpDataType_t,
                std::vector<int>& dims);
};

mluOpHandle_t mluOpGetCurrentHandle(c10::DeviceIndex device_index = -1);

class MluOpHandle {
 public:
  MluOpHandle() : handle(nullptr) { TORCH_MLUOP_CHECK(mluOpCreate(&handle)); }
  ~MluOpHandle() {
    if (handle) {
      TORCH_MLUOP_CHECK(mluOpDestroy(handle));
      handle = nullptr;
    }
  }
  void setQueue(cnrtQueue_t queue) {
    TORCH_MLUOP_CHECK(mluOpSetQueue(handle, queue));
  }
  mluOpHandle_t handle;
};

// modify tensor size and stride order based on
// channels_first to channels_last or channels_last_3d.
// which this is not same with pytorch original layout,
// this real layout is based on data storage real order.
// example: modify channels_last tensor dim to nhwc tensor desc.
//            N    C H W  -->   N    H W C
//          C*H*W  1 W C  --> C*H*W  W C 1
template <typename T>
void convertShapeAndStride(std::vector<T>& shape_info,
                           std::vector<T>& stride_info) {
  TORCH_MLU_CHECK(shape_info.size() == stride_info.size(),
                  "shape size need equal to stride size.");
  const int dim = shape_info.size();
  std::vector<T> temp_shape_info(dim);
  std::vector<T> temp_stride_info(dim);
  temp_shape_info[0] = shape_info[0];
  temp_stride_info[0] = stride_info[0];
  for (size_t i = 0; i < dim - 1; ++i) {
    const int index = (i + 1) % (dim - 1) + 1;
    temp_shape_info[i + 1] = shape_info[index];
    temp_stride_info[i + 1] = stride_info[index];
  }
  shape_info.assign(temp_shape_info.begin(), temp_shape_info.end());
  stride_info.assign(temp_stride_info.begin(), temp_stride_info.end());
}

// torch tensor provides int64_t type of shape and stride,
// but mluops descriptor requires type int32.
// use this function to ensure safe CAST, or report an error.
template <typename DST_T, typename SRC_T>
std::vector<DST_T> checkUpperBoundAndCastTo(const std::vector<SRC_T>& input) {
  std::vector<DST_T> output;
  output.reserve(input.size());
  for (const auto& val : input) {
    if (val > std::numeric_limits<DST_T>::max()) {
      TORCH_MLU_CHECK(false, "Requires dim size not greater than ",
                      std::numeric_limits<DST_T>::max(), ". But got ", val,
                      ".");
    }
    output.push_back(static_cast<DST_T>(val));
  }
  return output;
}
