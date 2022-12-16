#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


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
