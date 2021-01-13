#include <stdio.h>
#include <vector>

#include "common_cuda_helper.hpp"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
static int const threadsPerBlock = sizeof(unsigned long long int) * 8;
static const int MAXTENSORDIMS = 10;
struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
};

template <typename T>
__global__ void onnx_scatternd_kernel(const int n, const int* indices,
                                      const T* update, T* output,
                                      TensorDesc tensor_desc, int indice_cols) {
  const int copy_stride = tensor_desc.stride[indice_cols - 1];
  const int* shape = &(tensor_desc.shape[0]);
  const int* stride = &(tensor_desc.stride[0]);
  CUDA_1D_KERNEL_LOOP(index, n) {
    int output_offset = 0;
    const int* indices_current = indices + n * indice_cols;
    for (int i = 0; i < indice_cols; ++i) {
      output_offset += stride[i] * indices_current[i];
    }
    memcpy(output + output_offset, update + n * copy_stride,
           copy_stride * sizeof(T));
  }
}

template <typename T>
void TRTONNXScatterNDKernelLauncher(const T* data, const int* indices,
                                    const T* update, const int* dims,
                                    int nbDims, int indice_rows,
                                    int indice_cols, T* output,
                                    cudaStream_t stream) {
  // fill tensordesc and initial
  TensorDesc tensor_desc;
  memset((void*)&tensor_desc, 0, sizeof(TensorDesc));
  tensor_desc.shape[nbDims - 1] = dims[nbDims - 1];
  tensor_desc.stride[nbDims - 1] = 1;
  for (int i = nbDims - 2; i >= 0; --i) {
    tensor_desc.shape[i] = dims[i];
    tensor_desc.stride[i] = dims[i + 1] * tensor_desc.stride[i + 1];
  }
  const int data_size = tensor_desc.stride[0] * tensor_desc.shape[0];

  // output = np.copy(data)
  cudaMemcpyAsync(output, data, data_size * sizeof(T),
                  cudaMemcpyDeviceToDevice);

  // scatter
  const int col_block = DIVUP(indice_rows, threadsPerBlock);
  onnx_scatternd_kernel<<<col_block, threadsPerBlock, 0, stream>>>(
      indice_rows, indices, update, output, tensor_desc, indice_cols);
}

void TRTONNXScatterNDKernelLauncher_float(const float* data, const int* indices,
                                          const float* update, const int* dims,
                                          int nbDims, int indice_rows,
                                          int indice_cols, float* output,
                                          cudaStream_t stream) {
  TRTONNXScatterNDKernelLauncher<float>(data, indices, update, dims, nbDims,
                                        indice_rows, indice_cols, output,
                                        stream);
}

void TRTONNXScatterNDKernelLauncher_int32(const int* data, const int* indices,
                                          const int* update, const int* dims,
                                          int nbDims, int indice_rows,
                                          int indice_cols, int* output,
                                          cudaStream_t stream) {
  TRTONNXScatterNDKernelLauncher<int>(data, indices, update, dims, nbDims,
                                      indice_rows, indice_cols, output, stream);
}
