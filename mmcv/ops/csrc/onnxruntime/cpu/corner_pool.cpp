#include "corner_pool.h"
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

void BottomPoolForwardCPU(const float *input, float *output, float *tmp_output,
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
                // do bottom_pool
                for (int ind = 1; ind < height; ind <<= 1) {
                    for (int h = ind; h < height; h++) {
                        output[index_n_c + h * width + w] = std::max(tmp_output[h], tmp_output[h-ind]);
                    }
                    // copy column from updated output to tmp_output
                    for (int h = ind; h < height; h++) {
                        tmp_output[h] = output[index_n_c + h * width + w];
                    }
                } // for ind
            }  // for w
        } // for c
    }  // for n

}

void LeftPoolForwardCPU(const float *input, float *output, float *tmp_output,
                          const int nthreads, const int channels,
                          const int height, const int width) {
    int batch_size = nthreads / channels / width / height;
    for (int n = 0; n < batch_size; n++) {
        int index_n = n * channels * width * height;
        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * width * height;
            for (int h = 0; h < height; h++) {
                // copy row from output to tmp_output
                for (int w = 0; w < width; w++) {
                    int index = index_n_c + h * width + w;
                    tmp_output[w] = output[index];
                }
                // do left_pool
                for (int ind = 1; ind < width; ind <<= 1) {
                    for (int w = 0; w < width - ind; w++) {
                        output[index_n_c + h * width + w] = std::max(tmp_output[w], tmp_output[w+ind]);
                    }
                    // copy row from updated output to tmp_output
                    for (int w = 0; w < width - ind; w++) {
                        tmp_output[w] = output[index_n_c + h * width + w];
                    }
                } // for ind
            }  // for h
        } // for c
    }  // for n

}

void RightPoolForwardCPU(const float *input, float *output, float *tmp_output,
                          const int nthreads, const int channels,
                          const int height, const int width) {
    int batch_size = nthreads / channels / width / height;
    for (int n = 0; n < batch_size; n++) {
        int index_n = n * channels * width * height;
        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * width * height;
            for (int h = 0; h < height; h++) {
                // copy row from output to tmp_output
                for (int w = 0; w < width; w++) {
                    int index = index_n_c + h * width + w;
                    tmp_output[w] = output[index];
                }
                // do right_pool
                for (int ind = 1; ind < width; ind <<= 1) {
                    for (int w = ind; w < width; w++) {
                        output[index_n_c + h * width + w] = std::max(tmp_output[w], tmp_output[w-ind]);
                    }
                    // copy row from updated output to tmp_output
                    for (int w = ind; w < width; w++) {
                        tmp_output[w] = output[index_n_c + h * width + w];
                    }
                } // for ind
            }  // for h
        } // for c
    }  // for n

}

void  MMCVCornerPoolKernel::Compute(OrtKernelContext *context) {
    const int mode = int(mode_);
    typedef float T;
    const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
    const T *input_data = reinterpret_cast<const float *>(ort_.GetTensorData<T>(input));

    // allocate output memory
    OrtTensorDimensions out_dimensions(ort_, input);
    OrtValue *output = ort_.KernelContext_GetOutput(context, 0, out_dimensions.data(), out_dimensions.size());
    T *output_data = ort_.GetTensorMutableData<T>(output);

    // copy input_data to output_data
    int batch_size = out_dimensions.data()[0];
    int input_channels = out_dimensions.data()[1];
    int input_height = out_dimensions.data()[2];
    int input_width = out_dimensions.data()[3];
    int output_size = batch_size * input_channels * input_height * input_width;
    memcpy(output_data, input_data, sizeof(T) * output_size);

    // allocate tmp_output memory
    // 'top': 0, 'bottom': 1, 'left': 2, 'right':3
    assert(mode == 0 || mode == 1 || mode == 2 || mode == 3);
    int tmp_output_size;
    if (mode == 0 || mode_ == 1) tmp_output_size = input_height;
    else tmp_output_size = input_width;
    T *tmp_output_data = (T *)allocator_.Alloc(sizeof(T) * tmp_output_size);

    // do corner_pool
    if (mode == 0) TopPoolForwardCPU(input_data, output_data, tmp_output_data, output_size, input_channels, input_height, input_width);
    else if (mode == 1) BottomPoolForwardCPU(input_data, output_data, tmp_output_data, output_size, input_channels, input_height, input_width);
    else if (mode == 2) LeftPoolForwardCPU(input_data, output_data, tmp_output_data, output_size, input_channels, input_height, input_width);
    else RightPoolForwardCPU(input_data, output_data, tmp_output_data, output_size, input_channels, input_height, input_width);

}
