#ifndef _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_
#define _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include <diopi/diopirt.h>

namespace impl {

namespace aten {

#define TORCH_MM_VERSION (TORCH_VERSION_MAJOR * 1000 + TORCH_VERSION_MINOR * 10)
#define TORCH_1_7_MM_VERSION 1070
#define TORCH_1_8_MM_VERSION 1080
#define TORCH_1_9_MM_VERSION 1090
#define TORCH_1_10_MM_VERSION 1100
#define TORCH_1_11_MM_VERSION 1110
#define TORCH_1_12_MM_VERSION 1120

#define LOG_LINE_INFO() std::cerr << __FILE__ << ":" << __LINE__ << ": "; 

void logError(){std::cerr << std::endl;}

template<typename First, typename ...Rest>
void logError(First&& first, Rest&& ...rest) {
    std::cerr << std::forward<First>(first);
    logError(std::forward<Rest>(rest)...);
}

#define ATEN_NOT_SUPPORTED() \
    LOG_LINE_INFO() \
    logError("NotImplementError: function ", __FUNCTION__, " is not implemented for torch version ", \
        TORCH_VERSION);

#define ATEN_NOT_IMPLEMENT() \
    LOG_LINE_INFO() \
    logError("NotImplementError: function ", __FUNCTION__, " is not implemented for torch version ", \
        TORCH_VERSION);

#define NOT_SUPPORTED(str) \
    LOG_LINE_INFO() \
    logError("NotSupported: ", (str), ", ", __FILE__, ":", __LINE__);

using diopi_tensor_list = std::vector<diopiTensorHandle_t>;

inline void sync(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_handle));
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
    case  diopi_dtype_uint32:
        return caffe2::TypeMeta::Make<int32_t>();
    case diopi_dtype_int64:
    case diopi_dtype_uint64:
        return caffe2::TypeMeta::Make<int64_t>();
        return caffe2::TypeMeta::Make<uint64_t>();
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
    switch(input.scalar_type()) {
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

c10::DeviceType getATenDevice(diopiDevice_t device) {
    if (device == diopi_host) {
        return c10::DeviceType::CPU;
    } else if (device == diopi_device) {
        return c10::DeviceType::CUDA;
    } else {
        NOT_SUPPORTED("device dtype");
    }
}

at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes,
        at::IntArrayRef strides, const std::function<void(void*)>& deleter,
        at::Allocator* allocator, const at::TensorOptions& options) {
    auto device =
        at::globalContext().getDeviceFromPtr(data, options.device().type());
    if (options.device().has_index()) {
        assert(options.device() == device);
    }

    auto storage = at::Storage(
        at::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
        c10::InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
        allocator, false);
    return at::empty({0}, options).set_(storage, 0, sizes, strides);
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
    diopiGetTensorData(const_cast<diopiTensorHandle_t*>(&tensor), &data);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    at::IntArrayRef atDims(shape.data, shape.len);

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    at::IntArrayRef atStrides(stride.data, stride.len);

    auto options = at::TensorOptions(atDevice).dtype(atType);
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
    return isInt(scalar) ? scalar->ival : scalar->fval;
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

void updateATen2Tensor(diopiContextHandle_t ctx, const at::Tensor& atOut, diopiTensorHandle_t out) {
    at::Tensor atOutput = std::move(buildATen(out));
    atOutput.reshape_as(atOut).copy_(atOut, true);
    sync(ctx);
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
}

void buildDiopiTensor(diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t* out) {
    at::IntArrayRef atSize = input.sizes();
    at::IntArrayRef atStride = input.strides();
    diopiSize_t size(const_cast<int64_t*>(atSize.data()), atSize.size());
    diopiSize_t stride(const_cast<int64_t*>(atStride.data()), atStride.size());
    diopiDtype_t dtype = getDIOPITensorType(input);
    // 获取新的buffer
    // 是不是不可以这样子生成out？需要hacker input的内存到runtime中。
    diopiRequireTensor(ctx, out, &size, &stride, dtype, diopi_device);
    // diopiGetTensorData
    // (*out)->data_ptr
    // updateATen2Tensor 已更新 更新回out
    updateATen2Tensor(ctx, input, *out);

    // prefer method is: bool bRet = transferAcc(ctx, src, dst);
    // use directly the at storage

}

c10::optional<c10::string_view> getRoundingMode(diopiRoundMode_t rounding_mode) {
    switch(rounding_mode) {
    case (RoundModeNone): return "";
    case (RoundModeTrunc): return "trunc";
    case (RoundModeFloor): return "floor";
    case (RoundModeEND): return "";
    default: NOT_SUPPORTED("diopi round mode");
    }
}

}  // namespace aten

}  // namespace impl

#endif
