/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <diopi/diopirt.h>
#include <diopi_register.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>

// #include "helper.hpp"


extern "C" {

#define CALL_CUDA(Expr)   {                                                         \
    cudaError_t ret = Expr;                                                         \
    if (ret != cudaSuccess) {                                                       \
        printf("call a cudart function (%s) failed. return code=%d", #Expr, ret);   \
    }}


void* cuda_malloc(uint64_t bytes)
{
    void* ptr;
    CALL_CUDA(cudaMalloc(&ptr, bytes));
    return ptr;
}

void cuda_free(void* ptr)
{
    CALL_CUDA(cudaFree(ptr));
}

int32_t cuda_make_stream(diopiStreamHandle_t* stream_handle_ptr)
{
    cudaStream_t phStream;
    CALL_CUDA(cudaStreamCreate(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t cuda_destroy_stream(diopiStreamHandle_t stream_handle)
{
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaStreamDestroy(phStream));
    return diopiSuccess;
}

int32_t cuda_synchronize_stream(diopiStreamHandle_t stream_handle)
{
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaStreamSynchronize(phStream));
    return diopiSuccess;
}

int32_t cuda_memcpy_h2d_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes)
{
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, phStream));
    return diopiSuccess;
}

int32_t cuda_memcpy_d2h_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes)
{
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, phStream));
    return diopiSuccess;
}

int32_t cuda_memcpy_d2d_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes)
{
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, phStream));
    return diopiSuccess;
}

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

void set_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}

const char* cuda_get_last_error_string()
{
    cudaError_t error = cudaGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "cuda error: %s; other error: %s",
            cudaGetErrorString(error), strLastErrorOther);
    return strLastError;
}

int32_t initLibrary()
{
    diopiRegisterDeviceMallocFunc(cuda_malloc);
    diopiRegisterDevMemFreeFunc(cuda_free);
    diopiRegisterStreamCreateFunc(cuda_make_stream);
    diopiRegisterStreamDestroyFunc(cuda_destroy_stream);
    diopiRegisterSynchronizeStreamFunc(cuda_synchronize_stream);
    diopiRegisterMemcpyD2HAsyncFunc(cuda_memcpy_d2h_async);
    diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
    diopiRegisterMemcpyH2DAsyncFunc(cuda_memcpy_h2d_async);
    diopiRegisterGetLastErrorFunc(cuda_get_last_error_string);

    return diopiSuccess;
}

int32_t finalizeLibrary()
{
    return diopiSuccess;
}

}  // extern "C"
