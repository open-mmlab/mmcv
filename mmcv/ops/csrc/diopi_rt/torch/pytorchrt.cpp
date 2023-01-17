#include <diopi/diopirt.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <set>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "diopi.hpp"

extern "C" {

static char szVersion[256] = {0};

DIOPI_API const char* diopiGetVersion()
{
    return szVersion;
}

int32_t getitemsize(const diopiDtype_t dtype)
{
    switch (dtype)
    {
    case diopi_dtype_int32:
    case diopi_dtype_uint32:
    case diopi_dtype_float32:
    case diopi_dtype_tfloat32:
        return 4;
    case diopi_dtype_int64:
    case diopi_dtype_uint64:
    case diopi_dtype_float64:
        return 8;
    case diopi_dtype_int16:
    case diopi_dtype_uint16:
    case diopi_dtype_float16:
    case diopi_dtype_bfloat16:
        return 2;
    case diopi_dtype_int8:
    case diopi_dtype_uint8:
    case diopi_dtype_bool:
        return 1;
    default:
        assert(0);
    }
    return 0;
}

const char* diopi_dtype_to_str(const diopiDtype_t dtype)
{
#define _dtype2str(type) \
    if (type == dtype) return #type;
    _dtype2str(diopi_dtype_float16);
    _dtype2str(diopi_dtype_float32);
    _dtype2str(diopi_dtype_float64);
    _dtype2str(diopi_dtype_int8);
    _dtype2str(diopi_dtype_uint8);
    _dtype2str(diopi_dtype_int16);
    _dtype2str(diopi_dtype_uint16);
    _dtype2str(diopi_dtype_int32);
    _dtype2str(diopi_dtype_uint32);
    _dtype2str(diopi_dtype_int64);
    _dtype2str(diopi_dtype_uint64);
    _dtype2str(diopi_dtype_bool);
    _dtype2str(diopi_dtype_bfloat16);
    _dtype2str(diopi_dtype_tfloat32);

    return nullptr;
#undef _dtype2str
}

const char* device_to_str(const diopiDevice_t device)
{
#define _device2str(type) \
    if (type == device) return #type;
    _device2str(diopi_host);
    _device2str(diopi_device);

    return "Unknown device type\n";
#undef _device2str
}

diopiDtype_t scalartype2dtype(const c10::ScalarType dt) {
    switch (dt) {
        case c10::ScalarType::Byte: return diopi_dtype_uint8;
        case c10::ScalarType::Char: return diopi_dtype_int8;
        case c10::ScalarType::Short: return diopi_dtype_int16;
        case c10::ScalarType::Int: return diopi_dtype_int32;
        case c10::ScalarType::Long: return diopi_dtype_int64;
        case c10::ScalarType::Half: return diopi_dtype_float16;
        case c10::ScalarType::Float: return diopi_dtype_float32;
        case c10::ScalarType::Double: return diopi_dtype_float64;
        case c10::ScalarType::Bool: return diopi_dtype_bool;
        case c10::ScalarType::BFloat16: return diopi_dtype_bfloat16;
        default:
            std::cerr << "c10::ScalarType not supported in diopi";
    }
}

c10::ScalarType dtype2scalartype(const diopiDtype_t dt) {
    switch (dt) {
        case diopi_dtype_uint8: return c10::ScalarType::Byte;
        case diopi_dtype_int8: return c10::ScalarType::Char;
        case diopi_dtype_int16: return c10::ScalarType::Short;
        case diopi_dtype_int32:
        case diopi_dtype_uint32: return c10::ScalarType::Int;
        case diopi_dtype_int64:
        case diopi_dtype_uint64: return c10::ScalarType::Long;
        case diopi_dtype_float16: return c10::ScalarType::Half;
        case diopi_dtype_float32: return c10::ScalarType::Float;
        case diopi_dtype_float64: return c10::ScalarType::Double;
        case diopi_dtype_bool: return c10::ScalarType::Bool;
        case diopi_dtype_bfloat16: c10::ScalarType::BFloat16;
        default:
            std::cerr << "diopi dytpe not supported in pytorch+diopi scenario)";
    }
}

c10::DeviceType device2DeviceType(const diopiDevice_t device) {
    if (device == diopi_host) {
        return c10::DeviceType::CPU;
    } else if (device == diopi_device) {
        return c10::DeviceType::CUDA;
    } else {
        std::cerr << "device dtype not supported";
    }
}

#define CAST_TENSOR_HANDLE(th) reinterpret_cast<at::Tensor*>(th)

DIOPI_API diopiError_t diopiGetTensorData(diopiTensorHandle_t* th, void** pptr) {
    *pptr = CAST_TENSOR_HANDLE(*th)->data_ptr();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDataConst(const diopiTensorHandle_t* th, const void** pptr) {
    *pptr = CAST_TENSOR_HANDLE(*th)->data_ptr();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorShape(const diopiTensorHandle_t th, diopiSize_t* size) {
    at::IntArrayRef atShape = CAST_TENSOR_HANDLE(th)->sizes();
    *size = diopiSize_t(atShape.data(), atShape.size());
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorStride(const diopiTensorHandle_t th, diopiSize_t* stride) {
    at::IntArrayRef atStrides = CAST_TENSOR_HANDLE(th)->strides();
    *stride = diopiSize_t(atStrides.data(), atStrides.size());
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDtype(const diopiTensorHandle_t th, diopiDtype_t* dtype) {
    *dtype = scalartype2dtype(CAST_TENSOR_HANDLE(th)->scalar_type());
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDevice(const diopiTensorHandle_t th, diopiDevice_t* device) {
    *device = CAST_TENSOR_HANDLE(th)->is_cpu() ? diopi_host : diopi_device;
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorNumel(const diopiTensorHandle_t th, int64_t* numel) {
    *numel = CAST_TENSOR_HANDLE(th)->numel();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorElemSize(const diopiTensorHandle_t th, int64_t* itemsize) {
    diopiDtype_t dtype;
    auto ret = diopiGetTensorDtype(th, &dtype);
    if (ret != diopiSuccess) return ret;
    *itemsize = getitemsize(dtype);
    return diopiSuccess;
}

diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    *stream = at::cuda::getCurrentCUDAStream();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          const diopiSize_t* size, const diopiSize_t* stride,
                                          const diopiDtype_t dtype, const diopiDevice_t dev) {
    at::IntArrayRef atDims((*size).data, (*size).len);
    at::IntArrayRef atStrides((*stride).data, (*stride).len);
    auto options = at::TensorOptions(device2DeviceType(dev)).dtype(dtype2scalartype(dtype));
    ctx->arrays.push_back(at::empty(atDims, options));
    *tensor = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          int64_t bytes, diopiDevice_t dev) {
    diopiSize_t size(&bytes, 1);
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, dev);
}

}  // extern "C"
