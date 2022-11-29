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

int stack_vector_pool_forward_impl(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor support_features_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_batch_cnt_tensor, Tensor new_features_tensor,
    Tensor new_local_xyz_tensor, Tensor point_cnt_of_grid_tensor,
    Tensor grouped_idxs_tensor, const int num_grid_x, const int num_grid_y,
    const int num_grid_z, const float max_neighbour_distance, const int use_xyz,
    const int num_max_sum_points, const int nsample, const int neighbor_type,
    const int pooling_type) {
  return DISPATCH_DEVICE_IMPL(
      stack_vector_pool_forward_impl, support_xyz_tensor, xyz_batch_cnt_tensor,
      support_features_tensor, new_xyz_tensor, new_xyz_batch_cnt_tensor,
      new_features_tensor, new_local_xyz_tensor, point_cnt_of_grid_tensor,
      grouped_idxs_tensor, num_grid_x, num_grid_y, num_grid_z,
      max_neighbour_distance, use_xyz, num_max_sum_points, nsample,
      neighbor_type, pooling_type);
}

int stack_vector_pool_forward(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor support_features_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_batch_cnt_tensor, Tensor new_features_tensor,
    Tensor new_local_xyz_tensor, Tensor point_cnt_of_grid_tensor,
    Tensor grouped_idxs_tensor, const int num_grid_x, const int num_grid_y,
    const int num_grid_z, const float max_neighbour_distance, const int use_xyz,
    const int num_max_sum_points, const int nsample, const int neighbor_type,
    const int pooling_type) {
  return stack_vector_pool_forward_impl(
      support_xyz_tensor, xyz_batch_cnt_tensor, support_features_tensor,
      new_xyz_tensor, new_xyz_batch_cnt_tensor, new_features_tensor,
      new_local_xyz_tensor, point_cnt_of_grid_tensor, grouped_idxs_tensor,
      num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance, use_xyz,
      num_max_sum_points, nsample, neighbor_type, pooling_type);
}
void stack_vector_pool_backward_impl(const Tensor grad_new_features_tensor,
                                     const Tensor point_cnt_of_grid_tensor,
                                     const Tensor grouped_idxs_tensor,
                                     Tensor grad_support_features_tensor) {
  DISPATCH_DEVICE_IMPL(stack_vector_pool_backward_impl,
                       grad_new_features_tensor, point_cnt_of_grid_tensor,
                       grouped_idxs_tensor, grad_support_features_tensor);
}
void stack_vector_pool_backward(const Tensor grad_new_features_tensor,
                                const Tensor point_cnt_of_grid_tensor,
                                const Tensor grouped_idxs_tensor,
                                Tensor grad_support_features_tensor) {
  stack_vector_pool_backward_impl(grad_new_features_tensor,
                                  point_cnt_of_grid_tensor, grouped_idxs_tensor,
                                  grad_support_features_tensor);
}
