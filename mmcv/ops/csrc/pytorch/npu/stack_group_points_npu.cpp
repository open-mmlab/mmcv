using namespace NPU_NAME_SPACE;
using namespace std;

void stacK_group_points_forward_npu(int b, int c, int n, int nsample,
                                    const Tensor features_tensor,
                                    const Tensor features_batch_cnt_tensor,
                                    const Tensor idx_tensor,
                                    const Tensor idx_batch_cnt_tensor,
                                    Tensor out_tensor) {
  EXEC_NPU_CMD(aclnnStackGroupPoints, features_tensor,
               features_batch_cnt_tensor, idx_tensor, idx_batch_cnt_tensor,
               out_tensor);
}
 
void stack_group_points_forward_impl(int b, int c, int n, int nsample,
                                     const Tensor features_tensor,
                                     const Tensor features_batch_cnt_tensor,
                                     const Tensor idx_tensor,
                                     const Tensor idx_batch_cnt_tensor,
                                     Tensor out_tensor);
 
REGISTER_NPU_IMPL(stack_group_points_forward_impl,
                  stack_group_points_forward_npu);