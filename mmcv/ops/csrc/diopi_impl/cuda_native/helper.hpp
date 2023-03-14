/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef IMPL_CUDA_HELPER_HPP_
#define IMPL_CUDA_HELPER_HPP_

#include <diopi/diopirt.h>
#include <cuda_runtime.h>
#include <utility>
#include <assert.h>

#define DIOPI_CALL(Expr) {                                                              \
    diopiError_t ret = Expr;                                                            \
    if (diopiSuccess != ret) {                                                          \
        return ret;                                                                     \
    }}


namespace impl {

namespace cuda {

template<typename TensorType>
struct DataType;

template<>
struct DataType<diopiTensorHandle_t> {
    using type = void*;

    static void* data(diopiTensorHandle_t& tensor) {
        void *data;
        diopiGetTensorData(tensor, &data);
        return data;
    }
};

template<>
struct DataType<diopiConstTensorHandle_t> {
    using type = const void*;

    static const void* data(diopiConstTensorHandle_t& tensor) {
        const void *data;
        diopiGetTensorDataConst(tensor, &data);
        return data;
    }
};

template<typename TensorType>
class DiopiTensor final {
public:
    explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {}

    diopiDevice_t device() const {
        diopiDevice_t device;
        diopiGetTensorDevice(tensor_, &device);
        return device;
    }

    diopiDtype_t dtype() const {
        diopiDtype_t dtype;
        diopiGetTensorDtype(tensor_, &dtype);
        return dtype;
    }

    diopiDtype_t scalar_type() const {
        return dtype();
    }

    const diopiSize_t& shape() {
        diopiGetTensorShape(tensor_, &shape_);
        return shape_;
    }

    int64_t size(int i) {
        if (!shape_.data)
            diopiGetTensorShape(tensor_, &shape_);
        assert(i < shape_.len);
        const int64_t* shape_p = shape_.data;
        return *(shape_p + i);
    }

    int64_t ndimension() {
        if (!shape_.data)
            diopiGetTensorShape(tensor_, &shape_);
        return shape_.len;
    }

    const diopiSize_t& stride() {
        diopiGetTensorStride(tensor_, &stride_);
        return stride_;
    }

    int64_t numel() const {
        int64_t numel;
        diopiGetTensorNumel(tensor_, &numel);
        return numel;
    }
    int64_t elemsize() const {
        int64_t elemsize;
        diopiGetTensorElemSize(tensor_, &elemsize);
        return elemsize;
    }

    typename DataType<TensorType>::type data() {
        return DataType<TensorType>::data(tensor_);
    }

protected:
    TensorType tensor_;

    diopiSize_t shape_;
    diopiSize_t stride_;
};

template<typename TensorType>
auto makeTensor(TensorType& tensor) -> DiopiTensor<TensorType> {
    return DiopiTensor<TensorType>(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresTensor(
        diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return makeTensor(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresBuffer(
        diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor;
    diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
    return makeTensor(tensor);
}

inline cudaStream_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cudaStream_t>(stream_handle);
}

}  // namespace cuda

}  // namespace impl

#endif  // IMPL_CUDA_HELPER_HPP_
