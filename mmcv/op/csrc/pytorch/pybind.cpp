#include "pytorch_cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();

int carafe_naive_forward(
    Tensor features, Tensor masks, Tensor output,
    int kernel_size, int group_size, int scale_factor);

int carafe_naive_backward(
    Tensor top_grad, Tensor features, Tensor masks,
    Tensor bottom_grad, Tensor mask_grad,
    int kernel_size, int group_size, int scale_factor);

int carafe_forward(
    Tensor features, Tensor masks,
    Tensor rfeatures, Tensor routput,
    Tensor rmasks, Tensor output,
    int kernel_size, int group_size, int scale_factor);

int carafe_backward(
    Tensor top_grad, Tensor rfeatures,
    Tensor masks, Tensor rtop_grad,
    Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    int kernel_size, int group_size, int scale_factor);

int deform_conv_forward(
    Tensor input, Tensor weight,
    Tensor offset, Tensor output,
    Tensor columns, Tensor ones, int kW,
    int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step);

int deform_conv_backward_input(
    Tensor input, Tensor offset, Tensor gradOutput,
    Tensor gradInput, Tensor gradOffset, Tensor weight,
    Tensor columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int deformable_group, int im2col_step);

int deform_conv_backward_parameters(
    Tensor input, Tensor offset, Tensor gradOutput,
    Tensor gradWeight, Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step);

void deform_roi_pool_forward(
    Tensor input, Tensor rois, Tensor offset, Tensor output,
    int pooled_height, int pooled_width, float spatial_scale,
    int sampling_ratio, float gamma);

void deform_roi_pool_backward(
    Tensor grad_output, Tensor input, Tensor rois,
    Tensor offset, Tensor grad_input, Tensor grad_offset, int pooled_height,
    int pooled_width, float spatial_scale, int sampling_ratio,
    float gamma);

int sigmoid_focal_loss_forward(
    Tensor input, Tensor target, Tensor weight, Tensor output,
    float gamma, float alpha);

int sigmoid_focal_loss_backward(
    Tensor input, Tensor target, Tensor weight, Tensor grad_input,
    float gamma, float alpha);

int softmax_focal_loss_forward(
    Tensor input, Tensor target, Tensor weight, Tensor output,
    float gamma, float alpha);

int softmax_focal_loss_backward(
    Tensor input, Tensor target, Tensor weight, Tensor buff,
    Tensor grad_input, float gamma, float alpha);

void bbox_overlaps(
    const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
    const int mode, const bool aligned, const int offset);

int masked_im2col_forward(
    const Tensor im, const Tensor mask_h_idx, const Tensor mask_w_idx,
    Tensor col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w);

int masked_col2im_forward(
    const Tensor col, const Tensor mask_h_idx, const Tensor mask_w_idx,
    Tensor im, int height, int width, int channels);

int modulated_deform_conv_forward(
    Tensor input, Tensor weight,
    Tensor bias, Tensor ones,
    Tensor offset, Tensor mask,
    Tensor output, Tensor columns,
    int kernel_h, int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);

int modulated_deform_conv_backward(
    Tensor input, Tensor weight,
    Tensor bias, Tensor ones,
    Tensor offset, Tensor mask,
    Tensor columns,
    Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset,
    Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w, int group,
    int deformable_group, const bool with_bias);

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

int roi_align_forward(Tensor input, Tensor rois, Tensor output,
    Tensor argmax_y, Tensor argmax_x, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned);

int roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
    Tensor argmax_x, Tensor grad_input, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned);

int roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
    int pooled_height, int pooled_width, float spatial_scale);

int roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
    Tensor grad_input, int pooled_height, int pooled_width, float spatial_scale);

void syncbn_forward_step1(const Tensor input, Tensor mean,
         size_t n, size_t c, size_t h, size_t w);

void syncbn_forward_step2(const Tensor input, Tensor mean, Tensor var,
         size_t n, size_t c, size_t h, size_t w);

void syncbn_forward_step3(const Tensor input, Tensor mean, Tensor var,
         const Tensor weight, const Tensor bias, Tensor running_mean,
         Tensor running_var, Tensor std, Tensor output, 
         size_t n, size_t c, size_t h, size_t w, size_t group_size, const float eps,
         const float momentum);

void syncbn_backward_step1(const Tensor input, const Tensor mean,
         const Tensor std, const Tensor grad_output, Tensor weight_diff,
         Tensor bias_diff, size_t n, size_t c, size_t h, size_t w);

void syncbn_backward_step2(const Tensor input, const Tensor mean, 
         const Tensor weight, const Tensor weight_diff, const Tensor bias_diff,
         const Tensor std, const Tensor grad_output, Tensor grad_input, size_t n, size_t c, size_t h, size_t w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
    m.def("get_compiling_cuda_version", &get_compiling_cuda_version,
          "get_compiling_cuda_version");
    m.def("carafe_naive_forward", &carafe_naive_forward, "carafe_naive_forward",
          py::arg("features"), py::arg("masks"), py::arg("output"),
          py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
    m.def("carafe_naive_backward", &carafe_naive_backward, "carafe_naive_backward",
          py::arg("top_grad"), py::arg("features"), py::arg("masks"),
          py::arg("bottom_grad"), py::arg("mask_grad"),
          py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
    m.def("carafe_forward", &carafe_forward, "carafe_forward",
          py::arg("features"), py::arg("masks"), py::arg("rfeatures"),
          py::arg("routput"), py::arg("rmasks"), py::arg("output"),
          py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
    m.def("carafe_backward", &carafe_backward, "carafe_backward",
          py::arg("top_grad"), py::arg("rfeatures"), py::arg("masks"),
          py::arg("rtop_grad"), py::arg("rbottom_grad_hs"), py::arg("rbottom_grad"),
          py::arg("rmask_grad"), py::arg("bottom_grad"), py::arg("mask_grad"),
          py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
    m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward",
          py::arg("input"), py::arg("weight"), py::arg("offset"), py::arg("output"),
          py::arg("columns"), py::arg("ones"),
          py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
          py::arg("padH"), py::arg("padW"), py::arg("dilationW"), py::arg("dilationH"),
          py::arg("group"), py::arg("deformable_group"), py::arg("im2col_step"));
    m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input",
          py::arg("input"), py::arg("offset"), py::arg("gradOutput"), py::arg("gradInput"),
          py::arg("gradOffset"), py::arg("weight"), py::arg("columns"),
          py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
          py::arg("padH"), py::arg("padW"), py::arg("dilationW"), py::arg("dilationH"),
          py::arg("group"), py::arg("deformable_group"), py::arg("im2col_step"));
    m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters",
          py::arg("input"), py::arg("offset"), py::arg("gradOutput"), py::arg("gradWeight"),
          py::arg("columns"), py::arg("ones"),
          py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
          py::arg("padH"), py::arg("padW"), py::arg("dilationW"), py::arg("dilationH"),
          py::arg("group"), py::arg("deformable_group"), py::arg("scale"), py::arg("im2col_step"));
    m.def("deform_roi_pool_forward", &deform_roi_pool_forward, "deform roi pool forward",
          py::arg("input"), py::arg("rois"), py::arg("offset"), py::arg("output"),
          py::arg("pooled_height"), py::arg("pooled_width"),
          py::arg("spatial_scale"), py::arg("sampling_ratio"), py::arg("gamma"));
    m.def("deform_roi_pool_backward", &deform_roi_pool_backward, "deform roi pool backward",
          py::arg("grad_output"), py::arg("input"), py::arg("rois"), py::arg("offset"),
          py::arg("grad_input"), py::arg("grad_offset"),
          py::arg("pooled_height"), py::arg("pooled_width"),
          py::arg("spatial_scale"), py::arg("sampling_ratio"), py::arg("gamma"));
    m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward, "sigmoid_focal_loss_forward ",
          py::arg("input"), py::arg("target"), py::arg("weight"),
          py::arg("output"), py::arg("gamma"), py::arg("alpha"));
    m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward, "sigmoid_focal_loss_backward",
          py::arg("input"), py::arg("target"), py::arg("weight"),
          py::arg("grad_input"), py::arg("gamma"), py::arg("alpha"));
    m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward, "softmax_focal_loss_forward",
          py::arg("input"), py::arg("target"), py::arg("weight"),
          py::arg("output"), py::arg("gamma"), py::arg("alpha"));
    m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward, "softmax_focal_loss_backward",
          py::arg("input"), py::arg("target"), py::arg("weight"),
          py::arg("buff"), py::arg("grad_input"), py::arg("gamma"), py::arg("alpha"));
    m.def("bbox_overlaps", &bbox_overlaps, "bbox_overlaps",
          py::arg("bboxes1"), py::arg("bboxes2"), py::arg("ious"),
          py::arg("mode"), py::arg("aligned"), py::arg("offset"));
    m.def("masked_im2col_forward", &masked_im2col_forward, "masked_im2col_forward",
          py::arg("im"), py::arg("mask_h_idx"), py::arg("mask_w_idx"), 
          py::arg("col"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("pad_h"), py::arg("pad_w")); 
    m.def("masked_col2im_forward", &masked_col2im_forward, "masked_col2im_forward",
          py::arg("col"), py::arg("mask_h_idx"), py::arg("mask_w_idx"), 
          py::arg("im"), py::arg("height"), py::arg("width"), py::arg("channels"));
    m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward,
          "modulated deform conv forward",
          py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("ones"),
          py::arg("offset"), py::arg("mask"),
          py::arg("output"), py::arg("columns"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("dilation_h"), py::arg("dilation_w"), py::arg("group"),
          py::arg("deformable_group"), py::arg("with_bias"));
    m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward,
          "modulated deform conv backward",
          py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("ones"),
          py::arg("offset"), py::arg("mask"),
          py::arg("columns"),
          py::arg("grad_input"), py::arg("grad_weight"),
          py::arg("grad_bias"), py::arg("grad_offset"),
          py::arg("grad_mask"), py::arg("grad_output"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("dilation_h"), py::arg("dilation_w"), py::arg("group"),
          py::arg("deformable_group"), py::arg("with_bias"));
    m.def("nms", &nms, "nms (CPU/CUDA) ",
          py::arg("boxes"), py::arg("scores"),
          py::arg("iou_threshold"), py::arg("offset"));
    m.def("softnms", &softnms, "softnms (CPU) ",
          py::arg("boxes"), py::arg("scores"), py::arg("dets"),
          py::arg("iou_threshold"), py::arg("sigma"), py::arg("min_score"),
          py::arg("method"), py::arg("offset"));
    m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
          py::arg("input"), py::arg("rois"), py::arg("output"),
          py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
          py::arg("aligned_width"), py::arg("spatial_scale"),
          py::arg("sampling_ratio"), py::arg("pool_mode"),
          py::arg("aligned"));
    m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
          py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
          py::arg("argmax_x"), py::arg("grad_input"),
          py::arg("aligned_height"), py::arg("aligned_width"),
          py::arg("spatial_scale"), py::arg("sampling_ratio"),
          py::arg("pool_mode"), py::arg("aligned"));
    m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward",
          py::arg("input"), py::arg("rois"), py::arg("output"),
          py::arg("argmax"), py::arg("pooled_height"), py::arg("pooled_width"),
          py::arg("spatial_scale"));
    m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward",
          py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
          py::arg("grad_input"), py::arg("pooled_height"),
          py::arg("pooled_width"), py::arg("spatial_scale"));
    m.def("syncbn_forward_step1", &syncbn_forward_step1, "SyncBN forward_step1 (CUDA)",
          py::arg("input"), py::arg("mean"),
          py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"));
    m.def("syncbn_forward_step2", &syncbn_forward_step2, "SyncBN forward_step2 (CUDA)",
          py::arg("input"), py::arg("mean"), py::arg("var"),
          py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"));
    m.def("syncbn_forward_step3", &syncbn_forward_step3, "SyncBN forward_step3 (CUDA)",
          py::arg("input"), py::arg("mean"), py::arg("var"),
          py::arg("weight"), py::arg("bias"), py::arg("running_mean"),
          py::arg("running_var"), py::arg("std"), py::arg("output"),
          py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"),
          py::arg("group_size"), py::arg("eps"), py::arg("momentum"));
    m.def("syncbn_backward_step1", &syncbn_backward_step1, "SyncBN backward_step1 (CUDA)",
          py::arg("input"), py::arg("mean"), py::arg("std"),
          py::arg("grad_output"), py::arg("weight_diff"), py::arg("bias_diff"),
          py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"));
    m.def("syncbn_backward_step2", &syncbn_backward_step2, "SyncBN backward_step2 (CUDA)",
          py::arg("input"), py::arg("mean"), py::arg("weight"),
          py::arg("weight_diff"), py::arg("bias_diff"), py::arg("std"),
          py::arg("grad_output"), py::arg("grad_input"),
          py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"));
}
