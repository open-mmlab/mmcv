#ifndef _DIOPI_REFERENCE_IMPL_CAMB_ATEN_HPP_
#define _DIOPI_REFERENCE_IMPL_CAMB_ATEN_HPP_

// #include <ATen/ATen.h>
// #include <torch_mlu/torch_mlu.h>
#include "aten.h"
#include <cnnl.h>
#include <diopi/diopirt.h>

#include <utility>
#include <vector>
#include <iostream>
#include <exception>

#include <cnrt.h>
#include <mutex>

namespace camb {

namespace aten {

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

const char* camb_get_last_error_string() {
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "camb error: %s; other error: %s",
            ::cnrtGetErrorStr(err), strLastErrorOther);
    return strLastError;
}

void _set_last_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}
template<typename...Types>
void set_last_error_string(const char* szFmt, Types&& ...args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

#define ATEN_NOT_IMPLEMENT() \
    set_last_error_string("NotImplementError: function %s is not implemented for torch version %d" \
        __FUNCTION__, TORCH_VERSION); \

#define NOT_SUPPORTED(str) \
    set_last_error_string("NotSupported: %s at %s:%d", str, __FILE__, __LINE__); \

using diopi_tensor_list = std::vector<diopiTensorHandle_t>;

#define CALL_CAMB(Expr)   {                                                         \
    ::cnrtRet_t ret = Expr;                                                         \
    if (ret != ::CNRT_RET_SUCCESS) {                                                \
        printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret);   \
    }}                                                                              \


void sync(diopiContextHandle_t ctx) {
    CALL_CAMB(::cnrtSyncDevice());
    // diopiStreamHandle_t stream;
    // diopiGetStream(ctx, &stream);
    // cnrtQueue_t phStream = (cnrtQueue_t)stream;
    // ::cnrtQueueSync(phStream);
}

// Note: at::Tensor::to dosen't support int64/double, so we use the following functions to replace it.
inline at::Tensor toType(at::Tensor &atTarget, at::ScalarType dtype) {
    if (atTarget.scalar_type() != dtype) {
        auto atCpu = at::empty_like(atTarget, atTarget.options().device(at::kCPU));
        CALL_CAMB(::cnrtSyncDevice());
        ::cnrtMemcpy(atCpu.data_ptr(), atTarget.data_ptr(), atTarget.nbytes(), CNRT_MEM_TRANS_DIR_DEV2HOST);
        CALL_CAMB(::cnrtSyncDevice());
        auto atTmp = atCpu.to(dtype);
        auto atMlu = at::empty_like(atTmp, atTmp.options().device(at::kMLU));
        CALL_CAMB(::cnrtSyncDevice());
        ::cnrtMemcpy(atMlu.data_ptr(), atTmp.data_ptr(), atTmp.nbytes(), CNRT_MEM_TRANS_DIR_HOST2DEV);
        CALL_CAMB(::cnrtSyncDevice());
        atTarget = atMlu;
        return atMlu;
    }
    return atTarget;
}

void convertToRealLong(at::Tensor& src, at::Tensor& dst, at::ScalarType dtype) {
    if (dst.scalar_type() == dtype) {
        CALL_CAMB(::cnrtSyncDevice());
        ::cnrtMemcpy(dst.data_ptr(), src.data_ptr(), dst.nbytes(), CNRT_MEM_TRANS_DIR_DEV2DEV);
        CALL_CAMB(::cnrtSyncDevice());
    } else {
        // Note: convert int32 to int64 for indice, it's strange that returned indice with int64 type is int32 in real.
        auto atCpu = at::empty_like(src, src.options().device(at::kCPU)).to(dtype);
        CALL_CAMB(::cnrtSyncDevice());
        ::cnrtMemcpy(atCpu.data_ptr(), src.data_ptr(), atCpu.nbytes(), CNRT_MEM_TRANS_DIR_DEV2HOST);
        CALL_CAMB(::cnrtSyncDevice());
        auto atTmp = atCpu.contiguous().to(at::ScalarType::Long);
        CALL_CAMB(::cnrtSyncDevice());
        ::cnrtMemcpy(dst.data_ptr(), atTmp.data_ptr(), atTmp.nbytes(), CNRT_MEM_TRANS_DIR_HOST2DEV);
        CALL_CAMB(::cnrtSyncDevice());
    }
}

caffe2::TypeMeta getATenType(diopiDtype_t dt) {
    switch (dt) {
    case diopi_dtype_bool:
        return caffe2::TypeMeta::Make<bool>();
    case diopi_dtype_uint8:
        return caffe2::TypeMeta::Make<uint8_t>();
    case diopi_dtype_int8:
        return caffe2::TypeMeta::Make<int8_t>();
    case diopi_dtype_int16:
        return caffe2::TypeMeta::Make<int16_t>();
    case diopi_dtype_uint16:
        return caffe2::TypeMeta::Make<uint16_t>();
    case diopi_dtype_int32:
        return caffe2::TypeMeta::Make<int32_t>();
    case diopi_dtype_int64:
        return caffe2::TypeMeta::Make<int64_t>();
    case diopi_dtype_float32:
        return caffe2::TypeMeta::Make<float>();
    case diopi_dtype_float64:
        return caffe2::TypeMeta::Make<double>();
    case diopi_dtype_float16:
        return caffe2::TypeMeta::Make<at::Half>();
    case diopi_dtype_bfloat16:
        return caffe2::TypeMeta::Make<at::BFloat16>();
    default:
        NOT_SUPPORTED("diopi dytpe");
    }
}

diopiDtype_t getDIOPITensorType(at::Tensor& input) {
    switch (input.scalar_type()) {
    case at::ScalarType::Char:
        return diopi_dtype_bool;
    case at::ScalarType::Byte:
        return diopi_dtype_uint8;
    case at::ScalarType::Short:
        return diopi_dtype_int16;
    case at::ScalarType::Int:
        return diopi_dtype_int32;
    case at::ScalarType::Long:
        return diopi_dtype_int64;
    case at::ScalarType::Half:
        return diopi_dtype_float16;
    case at::ScalarType::Float:
        return diopi_dtype_float32;
    case at::ScalarType::Double:
        return diopi_dtype_float64;
    default:
        NOT_SUPPORTED("aten dtype");
    }
}

at::ScalarType getAtScalarType(diopiDtype_t type) {
    switch (type) {
    case diopi_dtype_bool:
        return at::ScalarType::Char;
    case diopi_dtype_uint8:
        return at::ScalarType::Byte;
    case diopi_dtype_int16:
        return at::ScalarType::Short;
    case diopi_dtype_int32:
        return at::ScalarType::Int;
    case diopi_dtype_int64:
        return at::ScalarType::Long;
    case diopi_dtype_float16:
        return at::ScalarType::Half;
    case diopi_dtype_float32:
        return at::ScalarType::Float;
    case diopi_dtype_float64:
        return at::ScalarType::Double;
    default:
        NOT_SUPPORTED("dioipi dtype");
    }
}

at::ScalarType getAtScalarType(diopiTensorHandle_t& tensor) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    return getAtScalarType(dtype);
}

c10::DeviceType getATenDevice(diopiDevice_t device) {
    if (device == diopi_host) {
        return c10::DeviceType::CPU;
    } else if (device == diopi_device) {
        return c10::DeviceType::MLU;
    } else {
        NOT_SUPPORTED("device dtype");
    }
}

at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes,
        at::IntArrayRef strides, const std::function<void(void*)>& deleter,
        at::Allocator* allocator, const at::TensorOptions& options) {
    auto device = c10::Device(c10::DeviceType::MLU, torch_mlu::current_device());
    if (options.device().has_index()) {
        assert(options.device() == device);
    }

    auto storage = at::Storage(
        at::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
        c10::InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
        allocator, false);
    auto tensor = torch_mlu::cnnl::ops::cnnl_empty({0}, options);
    return  torch_mlu::cnnl::ops::cnnl_set_(tensor, storage, 0, sizes, strides);
}

template<typename T>
at::Tensor buildATen(T tensor) {
    if (tensor == nullptr) return at::Tensor();
    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    caffe2::TypeMeta atType = getATenType(dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    c10::DeviceType atDevice = getATenDevice(device);
    void* data = nullptr;
    diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    at::IntArrayRef atDims(shape.data, shape.len);

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    at::IntArrayRef atStrides(stride.data, stride.len);

    auto options = at::TensorOptions(atDevice).dtype(atType).device(at::kMLU);
    int64_t numel = 0;
    diopiGetTensorNumel(tensor, &numel);
    if (0 == numel) {
        return at::empty(atDims, options);
    } else {
        at::Allocator* allocator = nullptr;
        return fromPreAllocated(data, atDims,
            atStrides, [](void*){}, allocator, options);
    }
}

inline bool isInt(const diopiScalar_t* scalar) {
    return scalar->stype <= 7;
}

inline bool isFloat(const diopiScalar_t* scalar) {
    return scalar->stype > 7;
}

inline at::Scalar buildAtScalar(const diopiScalar_t* scalar) {
    if (scalar == nullptr) {
        NOT_SUPPORTED("scalar is null ptr, we use temporarily zero");
        return at::Scalar();
    }
    // Note: indicate explictly the return type to make correctly at::Scalar.
    if ( isInt(scalar) ) {
        int64_t ival = scalar->ival;
        return ival;
    } else {
        double fval = scalar->fval;
        return fval;
    }
}

at::IntArrayRef buildAtIntArray(const diopiSize_t* size) {
    return at::IntArrayRef(size->data, size->len);
}

at::IntArrayRef buildAtIntArray(diopiSize_t size) {
    return at::IntArrayRef(size.data, size.len);
}

template<typename T>
decltype(auto) buildATenList(T* tensors, int64_t numTensors) {
    std::vector<at::Tensor> vecAtTensor;
    for (size_t i = 0; i < numTensors; ++i) {
        vecAtTensor.emplace_back(buildATen(tensors[i]));
    }
    return vecAtTensor;
}

void updateATen2Tensor(diopiContextHandle_t ctx, const at::Tensor& atSrc, diopiTensorHandle_t out) {
    at::Tensor atDst = buildATen(out);
    // Note: atDst.reshape_as(atSrc).copy_(atSrc); can not use copy_ for int64
    if (atDst.scalar_type() == at::ScalarType::Long || atDst.scalar_type() == at::ScalarType::Double) {
        if (atDst.is_mlu() && atSrc.is_mlu()) {
            CALL_CAMB(::cnrtSyncDevice());
            ::cnrtMemcpy(atDst.data_ptr(), atSrc.data_ptr(), atSrc.nbytes(), CNRT_MEM_TRANS_DIR_DEV2DEV);
            CALL_CAMB(::cnrtSyncDevice());
        } else {
            CALL_CAMB(::cnrtSyncDevice());
            ::cnrtMemcpy(atDst.data_ptr(), atSrc.data_ptr(), atSrc.nbytes(), CNRT_MEM_TRANS_DIR_HOST2DEV);
            CALL_CAMB(::cnrtSyncDevice());
        }
    } else {
        atDst.reshape_as(atSrc).copy_(atSrc, true);
    }
}

template<typename TupleT, std::size_t N>
struct UpdateTupleATen {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts,
            diopi_tensor_list& outs) {
        UpdateTupleATen<TupleT, N - 1>::update(ctx, atOuts, outs);
        updateATen2Tensor(ctx, std::get<N - 1>(atOuts), outs.at(N - 1));
    }
};

template<typename TupleT>
struct UpdateTupleATen<TupleT, 1> {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts,
            std::vector<diopiTensorHandle_t>& outs) {
        updateATen2Tensor(ctx, std::get<0>(atOuts), outs.at(0));
    }
};

template<typename TupleT>
void updateATen2Tensor(diopiContextHandle_t ctx, TupleT& atOuts, diopi_tensor_list& outs) {
    constexpr size_t tupleSize = std::tuple_size<TupleT>::value;
    UpdateTupleATen<TupleT, tupleSize>::update(ctx, atOuts, outs);
}

void updateATen2Tensor(diopiContextHandle_t ctx, std::vector<at::Tensor>& atOuts, diopi_tensor_list& outs) {
    for (size_t i = 0; i < atOuts.size(); ++i) {
        updateATen2Tensor(ctx, atOuts.at(i), outs.at(i));
    }
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopiTensorHandle_t out, Args&&... args) {
    at::Tensor atOut = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOut, out);
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopi_tensor_list& outs, Args&&... args) {
    auto atOuts = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOuts, outs);
}

template<typename Func, typename ...Args>
void invokeATenFuncInp(diopiContextHandle_t ctx, Func func, Args&&... args) {
    func(std::forward<Args>(args)...);
    sync(ctx);
}

void buildDiopiTensor(diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t* out) {
    at::IntArrayRef atSize = input.sizes();
    at::IntArrayRef atStride = input.strides();
    diopiSize_t size(const_cast<int64_t*>(atSize.data()), atSize.size());
    diopiSize_t stride(const_cast<int64_t*>(atStride.data()), atStride.size());
    diopiDtype_t dtype = getDIOPITensorType(input);
    diopiRequireTensor(ctx, out, &size, &stride, dtype, diopi_device);
    updateATen2Tensor(ctx, input, *out);
}

c10::optional<c10::string_view> getRoundingMode(diopiRoundMode_t rounding_mode) {
    switch (rounding_mode) {
    case (RoundModeNone): return "";
    case (RoundModeTrunc): return "trunc";
    case (RoundModeFloor): return "floor";
    case (RoundModeEND): return "";
    default: NOT_SUPPORTED("diopi round mode");
    }
}
}  // namespace aten

}  // namespace camb

#endif // _DIOPI_REFERENCE_IMPL_CAMB_ATEN_HPP_
