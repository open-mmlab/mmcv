#ifndef VECTOR_POOL_CUDA_KERNEL_CUH
#define VECTOR_POOL_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void query_stacked_local_neighbor_idxs_cuda_kernel(
    const T *support_xyz, const int *xyz_batch_cnt, const T *new_xyz, const int *new_xyz_batch_cnt,
    int *stack_neighbor_idxs, int *start_len, int *cumsum, int avg_length_of_neighbor_idxs,
    float max_neighbour_distance, int batch_size, int M, int nsample, int neighbor_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]
    // cumsum: (1), max offset of current data in stack_neighbor_idxs
    // max_neighbour_distance: float
    // nsample: find all (-1), find limited number(>0)
    // neighbor_type: 1: ball, others: cube

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < batch_size; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    support_xyz += xyz_batch_start_idx * 3;
    new_xyz += pt_idx * 3;
    start_len += pt_idx * 2;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    float local_x, local_y, local_z;
    float radius2 = max_neighbour_distance * max_neighbour_distance;

    int temp_idxs[1000];

    int sample_cnt = 0;
    for (int k = 0; k < n; ++k) {
        local_x = support_xyz[k * 3 + 0] - new_x;
        local_y = support_xyz[k * 3 + 1] - new_y;
        local_z = support_xyz[k * 3 + 2] - new_z;

        if (neighbor_type == 1){
            // ball
            if (local_x * local_x + local_y * local_y + local_z * local_z > radius2){
                continue;
            }
        }
        else{
            // voxel
            if ((fabs(local_x) > max_neighbour_distance) |
                (fabs(local_y) > max_neighbour_distance) |
                (fabs(local_z) > max_neighbour_distance)){
                continue;
            }
        }
        if (sample_cnt < 1000){
            temp_idxs[sample_cnt] = k;
        }
        else{
            break;
        }
        sample_cnt++;
        if (nsample > 0 && sample_cnt >= nsample) break;
    }
    start_len[0] = atomicAdd(cumsum, sample_cnt);
    start_len[1] = sample_cnt;

    int max_thresh = avg_length_of_neighbor_idxs * M;
    if (start_len[0] >= max_thresh) return;

    stack_neighbor_idxs += start_len[0];
    if (start_len[0] + sample_cnt >= max_thresh) sample_cnt = max_thresh - start_len[0];

    for (int k = 0; k < sample_cnt; k++){
        stack_neighbor_idxs[k] = temp_idxs[k] + xyz_batch_start_idx;
    }
}

template <typename T>
__global__ void query_three_nn_by_stacked_local_idxs_cuda_kernel(
    const T *support_xyz, const T *new_xyz, const T *new_xyz_grid_centers,
    int *new_xyz_grid_idxs, T *new_xyz_grid_dist2,
    const int *stack_neighbor_idxs, const int *start_len,
    int M, int num_total_grids){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
    // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
    // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]

    int grid_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= M || grid_idx >= num_total_grids) return;

    new_xyz += pt_idx * 3;
    new_xyz_grid_centers += pt_idx * num_total_grids * 3 + grid_idx * 3;
    new_xyz_grid_idxs += pt_idx * num_total_grids * 3 + grid_idx * 3;
    new_xyz_grid_dist2 += pt_idx * num_total_grids * 3 + grid_idx * 3;

    start_len += pt_idx * 2;
    stack_neighbor_idxs += start_len[0];
    int neighbor_length = start_len[1];

    float center_x = new_xyz_grid_centers[0];
    float center_y = new_xyz_grid_centers[1];
    float center_z = new_xyz_grid_centers[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = -1, besti2 = -1, besti3 = -1;
    for (int k = 0; k < neighbor_length; k++){
        int cur_neighbor_idx = stack_neighbor_idxs[k];

        float x = support_xyz[cur_neighbor_idx * 3 + 0];
        float y = support_xyz[cur_neighbor_idx * 3 + 1];
        float z = support_xyz[cur_neighbor_idx * 3 + 2];

        float d = (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) + (center_z - z) * (center_z - z);

        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = cur_neighbor_idx;
        }
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = cur_neighbor_idx;
        }
        else if (d < best3) {
            best3 = d; besti3 = cur_neighbor_idx;
        }
    }
    if (besti2 == -1){
        besti2 = besti1; best2 = best1;
    }
    if (besti3 == -1){
        besti3 = besti1; best3 = best1;
    }
    new_xyz_grid_dist2[0] = best1;
    new_xyz_grid_dist2[1] = best2;
    new_xyz_grid_dist2[2] = best3;
    new_xyz_grid_idxs[0] = besti1;
    new_xyz_grid_idxs[1] = besti2;
    new_xyz_grid_idxs[2] = besti3;
}

#endif
