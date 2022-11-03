#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void stack_query_local_neighbor_idxs_impl(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor new_xyz_tensor, const Tensor new_xyz_batch_cnt_tensor,
    Tensor stack_neighbor_idxs_tensor, Tensor start_len_tensor,
    Tensor cumsum_tensor, const int avg_length_of_neighbor_idxs,
    const float max_neighbour_distance, const int nsample,
    const int neighbor_type) {
  DISPATCH_DEVICE_IMPL(stack_query_local_neighbor_idxs_impl, support_xyz_tensor,
                       xyz_batch_cnt_tensor, new_xyz_tensor,
                       new_xyz_batch_cnt_tensor, stack_neighbor_idxs_tensor,
                       start_len_tensor, cumsum_tensor,
                       avg_length_of_neighbor_idxs, max_neighbour_distance,
                       nsample, neighbor_type);
}

void stack_query_local_neighbor_idxs(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor new_xyz_tensor, const Tensor new_xyz_batch_cnt_tensor,
    Tensor stack_neighbor_idxs_tensor, Tensor start_len_tensor,
    Tensor cumsum_tensor, const int avg_length_of_neighbor_idxs,
    const float max_neighbour_distance, const int nsample,
    const int neighbor_type) {
  stack_query_local_neighbor_idxs_impl(
      support_xyz_tensor, xyz_batch_cnt_tensor, new_xyz_tensor,
      new_xyz_batch_cnt_tensor, stack_neighbor_idxs_tensor, start_len_tensor,
      cumsum_tensor, avg_length_of_neighbor_idxs, max_neighbour_distance,
      nsample, neighbor_type);
}

void stack_query_three_nn_local_idxs_impl(
    const Tensor support_xyz_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_grid_centers_tensor, Tensor new_xyz_grid_idxs_tensor,
    Tensor new_xyz_grid_dist2_tensor, Tensor stack_neighbor_idxs_tensor,
    Tensor start_len_tensor, const int M, const int num_total_grids) {
  DISPATCH_DEVICE_IMPL(stack_query_three_nn_local_idxs_impl, support_xyz_tensor,
                       new_xyz_tensor, new_xyz_grid_centers_tensor,
                       new_xyz_grid_idxs_tensor, new_xyz_grid_dist2_tensor,
                       stack_neighbor_idxs_tensor, start_len_tensor, M,
                       num_total_grids);
}

void stack_query_three_nn_local_idxs(
    const Tensor support_xyz_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_grid_centers_tensor, Tensor new_xyz_grid_idxs_tensor,
    Tensor new_xyz_grid_dist2_tensor, Tensor stack_neighbor_idxs_tensor,
    Tensor start_len_tensor, const int M, const int num_total_grids) {
  stack_query_three_nn_local_idxs_impl(
      support_xyz_tensor, new_xyz_tensor, new_xyz_grid_centers_tensor,
      new_xyz_grid_idxs_tensor, new_xyz_grid_dist2_tensor,
      stack_neighbor_idxs_tensor, start_len_tensor, M, num_total_grids);
}
