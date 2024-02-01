/******************************************************************************
 * Copyright (c) 2022 Huawei Technologies Co., Ltd
 * All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MMCV_OPS_CSRC_COMMON_PYTORCH_NPU_UTIL_HPP_
#define MMCV_OPS_CSRC_COMMON_PYTORCH_NPU_UTIL_HPP_

#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <functional>
#include <type_traits>
#include <vector>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

#define NPU_NAME_SPACE at_npu::native

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef aclTensor *(*_aclCreateTensor)(
    const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
    const int64_t *stride, int64_t offset, aclFormat format,
    const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value,
                                               uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value,
                                               uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

#ifdef MMCV_WITH_XLA
#define DEVICE_TYPE at_npu::key::NativeDeviceType
#else
#define DEVICE_TYPE c10::DeviceType::PrivateUse1
#endif

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
  _(at::ScalarType::Byte, ACL_UINT8)                \
  _(at::ScalarType::Char, ACL_INT8)                 \
  _(at::ScalarType::Short, ACL_INT16)               \
  _(at::ScalarType::Int, ACL_INT32)                 \
  _(at::ScalarType::Long, ACL_INT64)                \
  _(at::ScalarType::Half, ACL_FLOAT16)              \
  _(at::ScalarType::Float, ACL_FLOAT)               \
  _(at::ScalarType::Double, ACL_DOUBLE)             \
  _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED)  \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)    \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)  \
  _(at::ScalarType::Bool, ACL_BOOL)                 \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::BFloat16, ACL_BF16)             \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)    \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

#define MEMCPY_TO_BUF(data_expression, size_expression)               \
  if (g_hashOffset + (size_expression) > kHashBufSize) {              \
    g_hashOffset = kHashBufMaxSize;                                   \
    return;                                                           \
  }                                                                   \
  memcpy(g_hashBuf + g_hashOffset, data_expression, size_expression); \
  g_hashOffset += size_expression;

inline const char *GetOpApiLibName(void) { return "libopapi.so"; }

inline const char *GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName,
                                   const char *apiName) {
  auto funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName,
                dlerror());
  }
  return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr) {
    ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
  }
  return handler;
}

inline void *GetOpApiFuncAddr(const char *apiName) {
  static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
  if (custOpApiHandler != nullptr) {
    auto funcAddr =
        GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor) {
  c10::Scalar expScalar;
  const at::Tensor *aclInput = &tensor;
  if (aclInput->scalar_type() == at::ScalarType::Double) {
    double value = *(double *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Long) {
    int64_t value = *(int64_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Float) {
    float value = *(float *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Int) {
    int value = *(int *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Half) {
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
    int8_t value = *(int8_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  }
  return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex),
                            cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar,
                                     at::ScalarType scalar_data_type) {
  return CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!at_tensor.defined()) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(
      acl_data_type != ACL_DT_UNDEFINED,
      std::string(c10::toString(scalar_data_type)) + " has not been supported")
  c10::SmallVector<int64_t, 5> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  auto itemsize = at_tensor.itemsize();
  if (itemsize == 0) {
    AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
    return nullptr;
  }
  if (acl_data_type != ACL_STRING) {
    storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
  }

  const auto dimNum = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    case 3:
      format = ACL_FORMAT_NCL;
      break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }

  if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
    at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
    return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(),
                           acl_data_type, aclInput.strides().data(),
                           aclInput.storage_offset(), format,
                           storageDims.data(), storageDims.size(),
                           const_cast<void *>(aclInput.storage().data()));
  }

  auto acl_tensor = aclCreateTensor(
      at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
      at_tensor.strides().data(), at_tensor.storage_offset(), format,
      storageDims.data(), storageDims.size(),
      const_cast<void *>(at_tensor.storage().data()));
  return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }

  at::ScalarType scalar_data_type = at_scalar.type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(
      acl_data_type != ACL_DT_UNDEFINED,
      std::string(c10::toString(scalar_data_type)) + " has not been supported")
  aclScalar *acl_scalar = nullptr;
  switch (scalar_data_type) {
    case at::ScalarType::Double: {
      double value = at_scalar.toDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Long: {
      int64_t value = at_scalar.toLong();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Bool: {
      bool value = at_scalar.toBool();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::ComplexDouble: {
      auto value = at_scalar.toComplexDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    default:
      acl_scalar = nullptr;
      break;
  }
  return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array) {
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
  for (size_t i = 0; i < at_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(at_tensor_list[i]);
  }
  auto acl_tensor_list =
      aclCreateTensorList(tensor_list.data(), tensor_list.size());
  return acl_tensor_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(opt_tensor.value());
  }
  return nullptr;
}

inline aclIntArray *ConvertType(
    const c10::optional<at::IntArrayRef> &opt_array) {
  if (opt_array.has_value()) {
    return ConvertType(opt_array.value());
  }
  return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar) {
  if (opt_scalar.has_value()) {
    return ConvertType(opt_scalar.value());
  }
  return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType) {
  return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

template <typename T>
T ConvertType(T value) {
  return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr,
                        std::index_sequence<I...>) {
  typedef int (*OpApiFunc)(
      typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr,
                            std::make_index_sequence<size>{});
}

inline void Release(aclTensor *p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }

  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }

  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p) {
  static const auto aclDestroyTensorList =
      GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &... args) {
  return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value) {
  MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value) {
  MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &... args) {
  AddParamToBuf(arg);
  AddParamToBuf(args...);
}

uint64_t CalcHashId();
typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);

#define EXEC_NPU_CMD(aclnn_api, ...)                                          \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      auto workspace_tensor =                                                 \
          at::empty({workspace_size}, options.dtype(kByte));                  \
      workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc =                                         \
          reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)

#endif  // MMCV_OPS_CSRC_COMMON_PYTORCH_NPU_UTIL_HPP_
