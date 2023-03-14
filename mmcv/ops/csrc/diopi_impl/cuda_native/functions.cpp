/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <diopi/functions.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <vector>
#include <mutex>

#include "helper.hpp"

extern "C" {
static char strLastErrorOther[4096] = {0};
static std::mutex mtxLastError;
}  // extern "C"

namespace impl {

namespace cuda {

void _set_last_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}

template<typename...Types>
void set_last_error_string(const char* szFmt, Types&&...args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

}  // namespace cuda

}  // namespace impl


#define DIOPI_CALLCUDNN(Expr) {                                                         \
        ::cudnnStatus_t ret = Expr;                                                     \
        if (CUDNN_STATUS_SUCCESS != ret) {                                              \
            impl::cuda::set_last_error_string("cudnn error %d : %s at %s:%s",           \
                    ret, cudnnGetErrorString(ret), __FILE__, __LINE__);                 \
            return diopiErrorOccurred;                                                  \
        }}                                                                              \

#define DIOPI_CHECKCUDNN(Expr) {                                                        \
        ::cudnnStatus_t ret = Expr;                                                     \
        if (CUDNN_STATUS_SUCCESS != ret) {                                              \
            impl::cuda::set_last_error_string("cudnn error %d : %s at %s:%s",           \
                    ret, cudnnGetErrorString(ret), __FILE__, __LINE__);                 \
        }}                                                                              \

static diopiError_t convertType(cudnnDataType_t *cudnnType, diopiDtype_t type) {
    switch (type) {
    case diopi_dtype_int8:
        *cudnnType = CUDNN_DATA_INT8;
        break;
    case diopi_dtype_uint8:
        *cudnnType = CUDNN_DATA_UINT8;
        break;
    case diopi_dtype_int32:
        *cudnnType = CUDNN_DATA_INT32;
        break;
    case diopi_dtype_float16:
        *cudnnType = CUDNN_DATA_HALF;
        break;
    case diopi_dtype_float32:
        *cudnnType = CUDNN_DATA_FLOAT;
        break;
    case diopi_dtype_float64:
        *cudnnType = CUDNN_DATA_DOUBLE;
        break;
#if CUDNN_VERSION >= 8400
    case diopi_dtype_bool:
        *cudnnType = CUDNN_DATA_BOOLEAN;
        break;
    case diopi_dtype_bfloat16:
        *cudnnType = CUDNN_DATA_BFLOAT16;
        break;
    case diopi_dtype_int64:
        *cudnnType = CUDNN_DATA_INT64;
        break;
#endif  // CUDNN_VERSION >= 8400
    default:
        impl::cuda::set_last_error_string("unknown diopitype error %d at %s:%s", type, __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}


namespace impl {

namespace cuda {

class CudnnScalar final {
public:
    template<typename T>
    void reset(const T& val) {
        if (data_) delete [] data_;
        data_ = new int8_t[sizeof(T)];
        T* ptr = reinterpret_cast<T*>(data_);
        *ptr = val;
    }

    void* data() const {
        return data_;
    }

    ~CudnnScalar() {
        if (data_) delete [] data_;
    }

protected:
    int8_t* data_{ nullptr };
};

template<typename T, cudnnStatus_t(*fnCreate)(T*), cudnnStatus_t(*fnDestroy)(T)>
class CudnnResourceGuard final {
public:
    CudnnResourceGuard() {
        DIOPI_CHECKCUDNN(fnCreate(&resource_));
    }

    ~CudnnResourceGuard() {
        DIOPI_CHECKCUDNN(fnDestroy(resource_));
    }

    T& get() {
        return resource_;
    }

protected:
    T resource_ {0};
};

diopiError_t setTensorDesc(diopiDtype_t type, const diopiSize_t& shape,
        const diopiSize_t& stride, cudnnTensorDescriptor_t desc) {
    cudnnDataType_t cudnnType;
    DIOPI_CALL(convertType(&cudnnType, type));

    int len = shape.len;
    int size = len < 4 ? 4 : len;
    std::vector<int> shapeArray(size);
    std::vector<int> strideArray(size);

    for (int i = 0; i < len; ++i) {
        shapeArray[i] = shape.data[i];
        strideArray[i] = stride.data[i];
    }
    for (int i = len; i < 4; ++i) {
        shapeArray[i] = 1;
        strideArray[i] = 1;
    }

    DIOPI_CALLCUDNN(cudnnSetTensorNdDescriptor(desc,
        cudnnType, size, shapeArray.data(), strideArray.data()));
    return diopiSuccess;
}

}  // namespace cuda

}  // namespace impl


extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                     diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    if (dim > 1) {
        impl::cuda::set_last_error_string("unknown dim error dim=%d at %s:%s", dim, __FILE__, __LINE__);
        return diopiErrorOccurred;
    }

    impl::cuda::CudnnResourceGuard<cudnnHandle_t, cudnnCreate, cudnnDestroy> handle;
    impl::cuda::CudnnResourceGuard<cudnnTensorDescriptor_t,
        cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor> desc;

    auto trIn = impl::cuda::makeTensor(input);
    auto trOut = impl::cuda::makeTensor(out);
    auto stream  = impl::cuda::getStream(ctx);
    if (0 == dim) {
        diopiSize_t oldShape = trIn.shape();
        diopiSize_t oldStride = trIn.stride();
        diopiSize_t newShape, newStride;
        int64_t len = oldShape.len + 1;
        std::vector<int64_t> shape(len);
        std::vector<int64_t> stride(len);
        shape[0] = 1;
        stride[0] = oldStride.data[0];
        for (int i = 0; i < oldShape.len; ++i) {
            shape[i + 1] = oldShape.data[i];
            stride[i + 1] = oldStride.data[i];
        }
        newShape.data = shape.data();
        newShape.len = len;
        newStride.data = stride.data();
        newStride.len = len;
        DIOPI_CALL(impl::cuda::setTensorDesc(trIn.dtype(), newShape, newStride, desc.get()));
    } else {
        DIOPI_CALL(impl::cuda::setTensorDesc(trIn.dtype(), trIn.shape(), trIn.stride(), desc.get()));
    }

    impl::cuda::CudnnScalar alpha, beta;
    if (dtype == diopi_dtype_float64) {
        alpha.reset<double>(1.0);
        beta.reset<double>(0.0);
    } else {
        alpha.reset<float>(1.f);
        beta.reset<float>(0.f);
    }
    DIOPI_CALLCUDNN(cudnnSetStream(handle.get(), stream));
    DIOPI_CALLCUDNN(cudnnSoftmaxForward(handle.get(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        alpha.data(), desc.get(), trIn.data(),
        beta.data(), desc.get(), trOut.data()));

    return diopiSuccess;
}

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::cuda::CudnnResourceGuard<cudnnHandle_t, cudnnCreate, cudnnDestroy> handle;
    impl::cuda::CudnnResourceGuard<cudnnTensorDescriptor_t,
        cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor> desc;
    impl::cuda::CudnnResourceGuard<cudnnActivationDescriptor_t,
        cudnnCreateActivationDescriptor, cudnnDestroyActivationDescriptor> descAct;

    auto trIn = impl::cuda::makeTensor(input);
    auto trOut = impl::cuda::makeTensor(out);
    auto stream = impl::cuda::getStream(ctx);

    DIOPI_CALL(impl::cuda::setTensorDesc(trIn.dtype(), trIn.shape(), trIn.stride(), desc.get()));
    DIOPI_CALLCUDNN(cudnnSetActivationDescriptor(descAct.get(), CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    impl::cuda::CudnnScalar alpha, beta;
    if (trIn.dtype() == diopi_dtype_float64) {
        alpha.reset<double>(1.0);
        beta.reset<double>(0.0);
    } else {
        alpha.reset<float>(1.f);
        beta.reset<float>(0.f);
    }
    DIOPI_CALLCUDNN(cudnnSetStream(handle.get(), stream));
    DIOPI_CALLCUDNN(cudnnActivationForward(handle.get(), descAct.get(), alpha.data(),
        desc.get(), trIn.data(), beta.data(), desc.get(), trOut.data()));

    return diopiSuccess;
}
