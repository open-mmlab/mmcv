#include "corner_pool.h"

#include <cmath>
#include <algorithm>

#include "../ort_mmcv_utils.h"

void TopPoolForwardCPU(const float *input, float *output, float *tmp_output,
                          const int nthreads, const int channels,
                          const int height, const int width) {
    int batch_size = nthreads / channels / width / height;
    for (int n = 0; n < batch_size; n++) {
        int index_n = n * channels * width * height;
        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * width * height;
            for (int w = 0; w < width; w++) {
                // copy column from output to tmp_output
                for (int h = 0; h < height; h++) {
                    int index = index_n_c + h * width + w;
                    tmp_output[h] = output[index];
                }
                // do top_pool
                for (int ind = 1; ind < height; ind <<= 1) {
                    for (int h = 0; h < height - ind; h++) {
                        output[index_n_c + h * width + w] = std::max(tmp_output[h], tmp_output[h+ind]);
                    }
                    // copy column from updated output to tmp_output
                    for (int h = 0; h < height - ind; h++) {
                        tmp_output[h] = output[index_n_c + h * width + w];
                    }
                } // for ind
            }  // for w
        } // for c
    }  // for n

}

void  MMCVTopPoolKernel::Compute(OrtKernelContext *context) {
    typedef float T;
    const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
    const T *input_data = reinterpret_cast<const float *>(ort_.GetTensorData<T>(input));

    OrtTensorDimensions out_dimensions(ort_, input);

    int input_channels = out_dimensions.data()[1];
    int input_height = out_dimensions.data()[2];
    int input_width = out_dimensions.data()[3];

    // allocate tmp and output memory
    OrtValue *output = ort_.KernelContext_GetOutput(context, 0, out_dimensions.data(), out_dimensions.size());
    T *output_data = ort_.GetTensorMutableData<T>(output);
    T *tmp_output_data = (T *)allocator_.Alloc(sizeof(T) * input_height);

    // copy input_data to output_data
    int output_size = out_dimensions.data()[0];
    for (auto i = 1; i < out_dimensions.size(); ++i) {
        output_size *= out_dimensions.data()[i];
    }
    memcpy(output_data, input_data, sizeof(T) * output_size);

    TopPoolForwardCPU(input_data, output_data, tmp_output_data, output_size, input_channels, input_height, input_width);
}
