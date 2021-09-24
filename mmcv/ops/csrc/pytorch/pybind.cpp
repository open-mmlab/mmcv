// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();

void carafe_naive_forward(Tensor features, Tensor masks, Tensor output,
                          int kernel_size, int group_size, int scale_factor);

void carafe_naive_backward(Tensor top_grad, Tensor features, Tensor masks,
                           Tensor bottom_grad, Tensor mask_grad,
                           int kernel_size, int group_size, int scale_factor);

void carafe_forward(Tensor features, Tensor masks, Tensor rfeatures,
                    Tensor routput, Tensor rmasks, Tensor output,
                    int kernel_size, int group_size, int scale_factor);

void carafe_backward(Tensor top_grad, Tensor rfeatures, Tensor masks,
                     Tensor rtop_grad, Tensor rbottom_grad_hs,
                     Tensor rbottom_grad, Tensor rmask_grad, Tensor bottom_grad,
                     Tensor mask_grad, int kernel_size, int group_size,
                     int scale_factor);

void deform_conv_forward(Tensor input, Tensor weight, Tensor offset,
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int dW, int dH, int padW, int padH,
                         int dilationW, int dilationH, int group,
                         int deformable_group, int im2col_step);

void deform_conv_backward_input(Tensor input, Tensor offset, Tensor gradOutput,
                                Tensor gradInput, Tensor gradOffset,
                                Tensor weight, Tensor columns, int kW, int kH,
                                int dW, int dH, int padW, int padH,
                                int dilationW, int dilationH, int group,
                                int deformable_group, int im2col_step);

void deform_conv_backward_parameters(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int dW, int dH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     int deformable_group, float scale,
                                     int im2col_step);

void deform_roi_pool_forward(Tensor input, Tensor rois, Tensor offset,
                             Tensor output, int pooled_height, int pooled_width,
                             float spatial_scale, int sampling_ratio,
                             float gamma);

void deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                              Tensor offset, Tensor grad_input,
                              Tensor grad_offset, int pooled_height,
                              int pooled_width, float spatial_scale,
                              int sampling_ratio, float gamma);

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha);

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset);

void masked_im2col_forward(const Tensor im, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor col,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w);

void masked_col2im_forward(const Tensor col, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor im, int height,
                           int width, int channels);

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);

void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, const int im2col_step);

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step);

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

std::vector<std::vector<int>> nms_match(Tensor dets, float iou_threshold);

std::vector<std::vector<float>> pixel_group(
    Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label,
    Tensor kernel_contour, int kernel_region_num, float distance_threshold);

std::vector<std::vector<int>> contour_expand(Tensor kernel_mask,
                                             Tensor internal_kernel_label,
                                             int min_kernel_area,
                                             int kernel_num);

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width, float spatial_scale);

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height, int pooled_width,
                       float spatial_scale);

void sync_bn_forward_mean(const Tensor input, Tensor mean);

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var);

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size);

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input);

void psamask_forward(const Tensor input, Tensor output, const int psa_type,
                     const int num_, const int h_feature, const int w_feature,
                     const int h_mask, const int w_mask, const int half_h_mask,
                     const int half_w_mask);

void psamask_backward(Tensor grad_output, const Tensor grad_input,
                      const int psa_type, const int num_, const int h_feature,
                      const int w_feature, const int h_mask, const int w_mask,
                      const int half_h_mask, const int half_w_mask);

void tin_shift_forward(Tensor input, Tensor shift, Tensor output);

void tin_shift_backward(Tensor grad_output, Tensor shift, Tensor grad_input);

void ball_query_forward(int b, int n, int m, float min_radius, float max_radius,
                        int nsample, Tensor new_xyz_tensor, Tensor xyz_tensor,
                        Tensor idx_tensor);

Tensor bottom_pool_forward(Tensor input);

Tensor bottom_pool_backward(Tensor input, Tensor grad_output);

Tensor left_pool_forward(Tensor input);

Tensor left_pool_backward(Tensor input, Tensor grad_output);

Tensor right_pool_forward(Tensor input);

Tensor right_pool_backward(Tensor input, Tensor grad_output);

Tensor top_pool_forward(Tensor input);

Tensor top_pool_backward(Tensor input, Tensor grad_output);

void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                     const int mode_flag, const bool aligned);

Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const float iou_threshold,
                   const int multi_label);

Tensor upfirdn2d(const Tensor &input, const Tensor &kernel, int up_x, int up_y,
                 int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0,
                 int pad_y1);

Tensor fused_bias_leakyrelu(const Tensor &input, const Tensor &bias,
                            const Tensor &refer, int act, int grad, float alpha,
                            float scale);

void roi_align_rotated_forward(Tensor input, Tensor rois, Tensor output,
                               int pooled_height, int pooled_width,
                               float spatial_scale, int sample_num,
                               bool aligned, bool clockwise);

void roi_align_rotated_backward(Tensor grad_output, Tensor rois,
                                Tensor grad_input, int pooled_height,
                                int pooled_width, float spatial_scale,
                                int sample_num, bool aligned, bool clockwise);

void border_align_forward(const Tensor &input, const Tensor &boxes,
                          Tensor output, Tensor argmax_idx,
                          const int pool_size);

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size);

void correlation_forward(Tensor input1, Tensor input2, Tensor output,
                         int kH, int kW, int patchH, int patchW,
                         int padH, int padW,
                         int dilationH, int dilationW,
                         int dilation_patchH, int dilation_patchW,
                         int dH, int dW);

void correlation_backward(Tensor grad_output,
                          Tensor input1, Tensor input2,
                          Tensor grad_input1, Tensor grad_input2,
                          int kH, int kW,
                          int patchH, int patchW,
                          int padH, int padW,
                          int dilationH, int dilationW,
                          int dilation_patchH, int dilation_patchW,
                          int dH, int dW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)", py::arg("input"),
        py::arg("kernel"), py::arg("up_x"), py::arg("up_y"), py::arg("down_x"),
        py::arg("down_y"), py::arg("pad_x0"), py::arg("pad_x1"),
        py::arg("pad_y0"), py::arg("pad_y1"));
  m.def("fused_bias_leakyrelu", &fused_bias_leakyrelu,
        "fused_bias_leakyrelu (CUDA)", py::arg("input"), py::arg("bias"),
        py::arg("empty"), py::arg("act"), py::arg("grad"), py::arg("alpha"),
        py::arg("scale"));
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_compiling_cuda_version", &get_compiling_cuda_version,
        "get_compiling_cuda_version");
  m.def("carafe_naive_forward", &carafe_naive_forward, "carafe_naive_forward",
        py::arg("features"), py::arg("masks"), py::arg("output"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_naive_backward", &carafe_naive_backward,
        "carafe_naive_backward", py::arg("top_grad"), py::arg("features"),
        py::arg("masks"), py::arg("bottom_grad"), py::arg("mask_grad"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_forward", &carafe_forward, "carafe_forward",
        py::arg("features"), py::arg("masks"), py::arg("rfeatures"),
        py::arg("routput"), py::arg("rmasks"), py::arg("output"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_backward", &carafe_backward, "carafe_backward",
        py::arg("top_grad"), py::arg("rfeatures"), py::arg("masks"),
        py::arg("rtop_grad"), py::arg("rbottom_grad_hs"),
        py::arg("rbottom_grad"), py::arg("rmask_grad"), py::arg("bottom_grad"),
        py::arg("mask_grad"), py::arg("kernel_size"), py::arg("group_size"),
        py::arg("scale_factor"));
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward",
        py::arg("input"), py::arg("weight"), py::arg("offset"),
        py::arg("output"), py::arg("columns"), py::arg("ones"), py::arg("kW"),
        py::arg("kH"), py::arg("dW"), py::arg("dH"), py::arg("padH"),
        py::arg("padW"), py::arg("dilationW"), py::arg("dilationH"),
        py::arg("group"), py::arg("deformable_group"), py::arg("im2col_step"));
  m.def("deform_conv_backward_input", &deform_conv_backward_input,
        "deform_conv_backward_input", py::arg("input"), py::arg("offset"),
        py::arg("gradOutput"), py::arg("gradInput"), py::arg("gradOffset"),
        py::arg("weight"), py::arg("columns"), py::arg("kW"), py::arg("kH"),
        py::arg("dW"), py::arg("dH"), py::arg("padH"), py::arg("padW"),
        py::arg("dilationW"), py::arg("dilationH"), py::arg("group"),
        py::arg("deformable_group"), py::arg("im2col_step"));
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters,
        "deform_conv_backward_parameters", py::arg("input"), py::arg("offset"),
        py::arg("gradOutput"), py::arg("gradWeight"), py::arg("columns"),
        py::arg("ones"), py::arg("kW"), py::arg("kH"), py::arg("dW"),
        py::arg("dH"), py::arg("padH"), py::arg("padW"), py::arg("dilationW"),
        py::arg("dilationH"), py::arg("group"), py::arg("deformable_group"),
        py::arg("scale"), py::arg("im2col_step"));
  m.def("deform_roi_pool_forward", &deform_roi_pool_forward,
        "deform roi pool forward", py::arg("input"), py::arg("rois"),
        py::arg("offset"), py::arg("output"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("gamma"));
  m.def("deform_roi_pool_backward", &deform_roi_pool_backward,
        "deform roi pool backward", py::arg("grad_output"), py::arg("input"),
        py::arg("rois"), py::arg("offset"), py::arg("grad_input"),
        py::arg("grad_offset"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("gamma"));
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward,
        "sigmoid_focal_loss_forward ", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward,
        "sigmoid_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("grad_input"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward,
        "softmax_focal_loss_forward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward,
        "softmax_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("buff"), py::arg("grad_input"),
        py::arg("gamma"), py::arg("alpha"));
  m.def("bbox_overlaps", &bbox_overlaps, "bbox_overlaps", py::arg("bboxes1"),
        py::arg("bboxes2"), py::arg("ious"), py::arg("mode"),
        py::arg("aligned"), py::arg("offset"));
  m.def("masked_im2col_forward", &masked_im2col_forward,
        "masked_im2col_forward", py::arg("im"), py::arg("mask_h_idx"),
        py::arg("mask_w_idx"), py::arg("col"), py::arg("kernel_h"),
        py::arg("kernel_w"), py::arg("pad_h"), py::arg("pad_w"));
  m.def("masked_col2im_forward", &masked_col2im_forward,
        "masked_col2im_forward", py::arg("col"), py::arg("mask_h_idx"),
        py::arg("mask_w_idx"), py::arg("im"), py::arg("height"),
        py::arg("width"), py::arg("channels"));
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward,
        "modulated deform conv forward", py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("output"), py::arg("columns"), py::arg("kernel_h"),
        py::arg("kernel_w"), py::arg("stride_h"), py::arg("stride_w"),
        py::arg("pad_h"), py::arg("pad_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("group"), py::arg("deformable_group"),
        py::arg("with_bias"));
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward,
        "modulated deform conv backward", py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("columns"), py::arg("grad_input"), py::arg("grad_weight"),
        py::arg("grad_bias"), py::arg("grad_offset"), py::arg("grad_mask"),
        py::arg("grad_output"), py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"),
        py::arg("pad_w"), py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("group"), py::arg("deformable_group"), py::arg("with_bias"));
  m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("scores"),
        py::arg("iou_threshold"), py::arg("offset"));
  m.def("softnms", &softnms, "softnms (CPU) ", py::arg("boxes"),
        py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
        py::arg("sigma"), py::arg("min_score"), py::arg("method"),
        py::arg("offset"));
  m.def("nms_match", &nms_match, "nms_match (CPU) ", py::arg("dets"),
        py::arg("iou_threshold"));
  m.def("pixel_group", &pixel_group, "pixel group (CPU) ", py::arg("score"),
        py::arg("mask"), py::arg("embedding"), py::arg("kernel_label"),
        py::arg("kernel_contour"), py::arg("kernel_region_label"),
        py::arg("distance_threshold"));
  m.def("contour_expand", &contour_expand, "contour exapnd (CPU) ",
        py::arg("kernel_mask"), py::arg("internal_kernel_label"),
        py::arg("min_kernel_area"), py::arg("kernel_num"));
  m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"), py::arg("argmax"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
        py::arg("grad_input"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"));
  m.def("sync_bn_forward_mean", &sync_bn_forward_mean, "sync_bn forward_mean",
        py::arg("input"), py::arg("mean"));
  m.def("sync_bn_forward_var", &sync_bn_forward_var, "sync_bn forward_var",
        py::arg("input"), py::arg("mean"), py::arg("var"));
  m.def("sync_bn_forward_output", &sync_bn_forward_output,
        "sync_bn forward_output", py::arg("input"), py::arg("mean"),
        py::arg("var"), py::arg("weight"), py::arg("bias"),
        py::arg("running_mean"), py::arg("running_var"), py::arg("norm"),
        py::arg("std"), py::arg("output"), py::arg("eps"), py::arg("momentum"),
        py::arg("group_size"));
  m.def("sync_bn_backward_param", &sync_bn_backward_param,
        "sync_bn backward_param", py::arg("grad_output"), py::arg("norm"),
        py::arg("grad_weight"), py::arg("grad_bias"));
  m.def("sync_bn_backward_data", &sync_bn_backward_data,
        "sync_bn backward_data", py::arg("grad_output"), py::arg("weight"),
        py::arg("grad_weight"), py::arg("grad_bias"), py::arg("norm"),
        py::arg("std"), py::arg("grad_input"));
  m.def("psamask_forward", &psamask_forward, "PSAMASK forward (CPU/CUDA)",
        py::arg("input"), py::arg("output"), py::arg("psa_type"),
        py::arg("num_"), py::arg("h_feature"), py::arg("w_feature"),
        py::arg("h_mask"), py::arg("w_mask"), py::arg("half_h_mask"),
        py::arg("half_w_mask"));
  m.def("psamask_backward", &psamask_backward, "PSAMASK backward (CPU/CUDA)",
        py::arg("grad_output"), py::arg("grad_input"), py::arg("psa_type"),
        py::arg("num_"), py::arg("h_feature"), py::arg("w_feature"),
        py::arg("h_mask"), py::arg("w_mask"), py::arg("half_h_mask"),
        py::arg("half_w_mask"));
  m.def("tin_shift_forward", &tin_shift_forward, "tin_shift forward",
        py::arg("input"), py::arg("shift"), py::arg("output"));
  m.def("tin_shift_backward", &tin_shift_backward, "tin_shift backward",
        py::arg("grad_output"), py::arg("shift"), py::arg("grad_input"));
  m.def("bottom_pool_forward", &bottom_pool_forward, "Bottom Pool Forward",
        py::arg("input"), py::call_guard<py::gil_scoped_release>());
  m.def("bottom_pool_backward", &bottom_pool_backward, "Bottom Pool Backward",
        py::arg("input"), py::arg("grad_output"),
        py::call_guard<py::gil_scoped_release>());
  m.def("left_pool_forward", &left_pool_forward, "Left Pool Forward",
        py::arg("input"), py::call_guard<py::gil_scoped_release>());
  m.def("left_pool_backward", &left_pool_backward, "Left Pool Backward",
        py::arg("input"), py::arg("grad_output"),
        py::call_guard<py::gil_scoped_release>());
  m.def("right_pool_forward", &right_pool_forward, "Right Pool Forward",
        py::arg("input"), py::call_guard<py::gil_scoped_release>());
  m.def("right_pool_backward", &right_pool_backward, "Right Pool Backward",
        py::arg("input"), py::arg("grad_output"),
        py::call_guard<py::gil_scoped_release>());
  m.def("top_pool_forward", &top_pool_forward, "Top Pool Forward",
        py::arg("input"), py::call_guard<py::gil_scoped_release>());
  m.def("top_pool_backward", &top_pool_backward, "Top Pool Backward",
        py::arg("input"), py::arg("grad_output"),
        py::call_guard<py::gil_scoped_release>());
  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes",
        py::arg("boxes1"), py::arg("boxes2"), py::arg("ious"),
        py::arg("mode_flag"), py::arg("aligned"));
  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes", py::arg("dets"),
        py::arg("scores"), py::arg("order"), py::arg("dets_sorted"),
        py::arg("iou_threshold"), py::arg("multi_label"));
  m.def("ball_query_forward", &ball_query_forward, "ball_query_forward",
        py::arg("b"), py::arg("n"), py::arg("m"), py::arg("min_radius"),
        py::arg("max_radius"), py::arg("nsample"), py::arg("new_xyz_tensor"),
        py::arg("xyz_tensor"), py::arg("idx_tensor"));
  m.def("roi_align_rotated_forward", &roi_align_rotated_forward,
        "roi_align_rotated forward", py::arg("input"), py::arg("rois"),
        py::arg("output"), py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"), py::arg("sample_num"), py::arg("aligned"),
        py::arg("clockwise"));
  m.def("roi_align_rotated_backward", &roi_align_rotated_backward,
        "roi_align_rotated backward", py::arg("grad_output"), py::arg("rois"),
        py::arg("grad_input"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sample_num"), py::arg("aligned"), py::arg("clockwise"));
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward,
        "forward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("im2col_step"));
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward,
        "backward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("im2col_step"));
  m.def("border_align_forward", &border_align_forward,
        "forward function of border_align", py::arg("input"), py::arg("boxes"),
        py::arg("output"), py::arg("argmax_idx"), py::arg("pool_size"));
  m.def("border_align_backward", &border_align_backward,
        "backward function of border_align", py::arg("grad_output"),
        py::arg("boxes"), py::arg("argmax_idx"), py::arg("grad_input"),
        py::arg("pool_size"));
  m.def("correlation_forward", &correlation_forward, "Correlation forward");
  m.def("correlation_backward", &correlation_backward, "Correlation backward");
}
