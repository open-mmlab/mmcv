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

diopiDtype_t scalartype2dtype(c10::ScalarType dt) {
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
    // ctx->getStreamHandle();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          const diopiSize_t* size, const diopiSize_t* stride,
                                          const diopiDtype_t dtype, const diopiDevice_t dev) {
    // ctx->createTensor
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          int64_t bytes, diopiDevice_t dev) {
    diopiSize_t size(&bytes, 1);
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, dev);
}

}  // extern "C"
