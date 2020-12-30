#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/gather.h>

#include <chrono>
#include <thread>
#include <vector>

#include "common_cuda_helper.hpp"
#include "nms_cuda_kernel.cuh"
#include "trt_cuda_helper.cuh"
struct NMSBox {
  float box[4];
};

struct nm_centerwh2xyxy {
  __host__ __device__ NMSBox operator()(const NMSBox box) {
    NMSBox out;
    out.box[0] = box.box[0] - box.box[2] / 2.0f;
    out.box[1] = box.box[1] - box.box[3] / 2.0f;
    out.box[2] = box.box[0] + box.box[2] / 2.0f;
    out.box[3] = box.box[1] + box.box[3] / 2.0f;
    return out;
  }
};

struct nm_sbox_idle {
  __host__ __device__ NMSBox operator()(const NMSBox box) {
    return {0.0f, 0.0f, 1.0f, 1.0f};
  }
};

struct nms_score_threshold {
  float score_threshold_;
  __host__ __device__ nms_score_threshold(const float* score_threshold) {
    score_threshold_ = *score_threshold;
  }

  __host__ __device__ bool operator()(const float score) {
    return score < score_threshold_;
  }
};

__global__ void nms_reindex_kernel(int n, int* output, int* index_cache){
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int old_index = output[index*3+2];
    output[index*3+2] = index_cache[old_index];
  }
}

size_t get_onnxnms_workspace_size(size_t num_batches, size_t spatial_dimension,
                                  size_t num_classes, size_t boxes_word_size,
                                  int center_point_box) {
  size_t boxes_xyxy_workspace = 0;
  if (center_point_box == 1) {
    boxes_xyxy_workspace =
        num_batches * spatial_dimension * 4 * boxes_word_size;
  }
  size_t scores_workspace = spatial_dimension * boxes_word_size;
  size_t boxes_workspace =
      num_batches * spatial_dimension * 4 * boxes_word_size;
  const int col_blocks = DIVUP(spatial_dimension, threadsPerBlock);
  size_t mask_workspace =
      num_batches * spatial_dimension * col_blocks * sizeof(unsigned long long);
  size_t index_workspace =
      spatial_dimension * (num_batches * num_classes + 1) * sizeof(int);
  return scores_workspace + boxes_xyxy_workspace + boxes_workspace +
         mask_workspace + index_workspace;
}

void TRTONNXNMSCUDAKernelLauncher_float(const float* boxes, const float* scores,
                                        const int* max_output_boxes_per_class,
                                        const float* iou_threshold,
                                        const float* score_threshold,
                                        int* output, int center_point_box,
                                        int num_batches, int spatial_dimension,
                                        int num_classes, void* workspace,
                                        cudaStream_t stream) {
  const int col_blocks = DIVUP(spatial_dimension, threadsPerBlock);
  float* boxes_sorted = (float*)workspace;
  workspace = workspace + num_batches * spatial_dimension * 4 * sizeof(float);

  float* boxes_xyxy = nullptr;
  if (center_point_box == 1) {
    boxes_xyxy = (float*)workspace;
    workspace = workspace + num_batches * spatial_dimension * 4 * sizeof(float);
    thrust::transform(thrust::cuda::par.on(stream), (NMSBox*)boxes,
                      (NMSBox*)(boxes + num_batches * spatial_dimension * 4),
                      (NMSBox*)boxes_xyxy, nm_centerwh2xyxy());
    cudaCheckError();
  }

  float* scores_sorted = (float*)workspace;
  workspace = workspace + spatial_dimension * sizeof(float);

  unsigned long long* dev_mask = (unsigned long long*)workspace;
  workspace = workspace + num_batches * spatial_dimension * col_blocks *
                              sizeof(unsigned long long);

  // generate sequence [0,1,2,3,4 ....]
  int* index_template = (int*)workspace;
  workspace = workspace + spatial_dimension * sizeof(int);
  thrust::sequence(thrust::cuda::par.on(stream), index_template,
                   index_template + spatial_dimension, 0);

  int* index_cache = (int*)workspace;
  workspace =
      workspace + spatial_dimension * num_batches * num_classes * sizeof(int);

  // this is dirty, should we create another nms_cuda kernel with float*
  // iou_threshold?
  float iou_threshold_cpu = 0.0f;
  if (iou_threshold != nullptr) {
    cudaMemcpy(&iou_threshold_cpu, iou_threshold, sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaCheckError();
  }

  int max_output_boxes_per_class_cpu = 0;
  if (max_output_boxes_per_class != nullptr) {
    cudaMemcpy(&max_output_boxes_per_class_cpu, max_output_boxes_per_class,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
  }
  if (max_output_boxes_per_class_cpu == 0) {
    max_output_boxes_per_class_cpu = spatial_dimension;
  }

  // allocate pined memory
  int* output_cpu = nullptr;
  cudaMallocHost((void**)&output_cpu, num_batches * spatial_dimension *
                                          num_classes * 3 * sizeof(int));
  cudaCheckError();
  memset(output_cpu, 0,
         num_batches * spatial_dimension * num_classes * sizeof(int));
  unsigned long long* dev_mask_cpu = nullptr;
  cudaMallocHost((void**)&dev_mask_cpu, num_batches * spatial_dimension *
                                            col_blocks *
                                            sizeof(unsigned long long));
  cudaCheckError();

  std::vector<unsigned long long> remv(col_blocks);

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  const int offset = 1;

  int output_count = 0;
  int* output_cpu_current = output_cpu;
  std::vector<int> out_batch_ids(num_batches);
  std::vector<int> out_cls_ids(num_classes);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    for (int cls_id = 0; cls_id < num_classes; ++cls_id) {
      const int batch_cls_id = batch_id * num_classes + cls_id;
      const int process_id = batch_cls_id % num_batches;
      out_batch_ids[process_id] = batch_id;
      out_cls_ids[process_id] = cls_id;
      float* boxes_sorted_current =
          boxes_sorted + process_id * spatial_dimension * 4;
      int* index_current = index_cache + batch_cls_id * spatial_dimension;

      // sort boxes by score
      cudaMemcpyAsync(scores_sorted, scores + batch_cls_id * spatial_dimension,
                      spatial_dimension * sizeof(float),
                      cudaMemcpyDeviceToDevice, stream);
      cudaCheckError();

      cudaMemcpyAsync(index_current, index_template,
                      spatial_dimension * sizeof(int), cudaMemcpyDeviceToDevice,
                      stream);

      thrust::sort_by_key(thrust::cuda::par.on(stream), scores_sorted,
                          scores_sorted + spatial_dimension,
                          index_current,
                          thrust::greater<float>());

      if (center_point_box == 1) {
      thrust::gather(thrust::cuda::par.on(stream), index_current, index_current+spatial_dimension,
      (NMSBox*)(boxes_xyxy + batch_id * spatial_dimension * 4), (NMSBox*)boxes_sorted_current
      );
      }else{
      thrust::gather(thrust::cuda::par.on(stream), index_current, index_current+spatial_dimension,
      (NMSBox*)(boxes + batch_id * spatial_dimension * 4), (NMSBox*)boxes_sorted_current
      );

      }

      cudaCheckError();

      // if (score_threshold != nullptr) {
      //   thrust::transform_if(
      //       thrust::cuda::par.on(stream), (NMSBox*)boxes_sorted_current,
      //       (NMSBox*)(boxes_sorted_current + spatial_dimension * 4),
      //       scores_sorted, (NMSBox*)boxes_sorted_current, nm_sbox_idle(),
      //       nms_score_threshold(score_threshold));
      // }

      unsigned long long* dev_mask_current =
          dev_mask + batch_id * spatial_dimension * col_blocks;

      nms_cuda<<<blocks, threads, 0, stream>>>(
          spatial_dimension, iou_threshold_cpu, offset, boxes_sorted_current,
          dev_mask_current);

      // process cpu
      if (process_id == num_batches - 1) {
        cudaMemcpy(dev_mask_cpu, dev_mask,
                   num_batches * spatial_dimension * col_blocks *
                       sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        cudaCheckError();
        for (int mask_batch_id = 0; mask_batch_id < num_batches;
             ++mask_batch_id) {
          const int out_batch_id = out_batch_ids[mask_batch_id];
          const int out_cls_id = out_cls_ids[mask_batch_id];
          unsigned long long* dev_mask_cpu_current =
              dev_mask_cpu + mask_batch_id * spatial_dimension * col_blocks;

          int index_offset = out_batch_id*num_classes*spatial_dimension + out_cls_id*spatial_dimension;

          memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
          int out_per_class_count = 0;
          for (int i = 0; i < spatial_dimension; i++) {
            int nblock = i / threadsPerBlock;
            int inblock = i % threadsPerBlock;
            if (!(remv[nblock] & (1ULL << inblock))) {
              output_cpu_current[0] = out_batch_id;
              output_cpu_current[1] = out_cls_id;
              output_cpu_current[2] = index_offset+i;
              output_cpu_current += 3;
              output_count += 1;
              out_per_class_count += 1;
              if (out_per_class_count >= max_output_boxes_per_class_cpu) {
                break;
              }
              // set every overlap box with bit 1 in remv
              unsigned long long* p = dev_mask_cpu_current + i * col_blocks;
              for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
              }  // j
            }
          }  // i
        }    // mask_batch_id
      }
    }  // cls_id
  }    // batch_id

  // fill output with -1
  thrust::fill(thrust::cuda::par.on(stream), output,
               output + num_batches * spatial_dimension * num_classes, -1);
  cudaCheckError();

  cudaMemcpy(output, output_cpu, size_t(output_count * 3 * sizeof(int)),
             cudaMemcpyDefault);
  cudaCheckError();

nms_reindex_kernel<<<DIVUP(output_count, threadsPerBlock), threadsPerBlock, 0, stream>>>(output_count, output, index_cache);

  // free pined memory
  cudaFreeHost(dev_mask_cpu);
  cudaFreeHost(output_cpu);
}
