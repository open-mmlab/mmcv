#include "parrots_cpp_helper.hpp"

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

void psamask_collect_forward(const int num_, const int h_feature,
                             const int w_feature, const int h_mask,
                             const int w_mask, const int half_h_mask,
                             const int half_w_mask, const float *mask_data,
                             float *buffer_data) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            buffer_data[(n * h_feature * w_feature +
                         (hidx + h - half_h_mask) * w_feature +
                         (widx + w - half_w_mask)) *
                            h_feature * w_feature +
                        h * w_feature + w] =
                mask_data[((n * h_mask * w_mask + hidx * w_mask + widx) *
                               h_feature +
                           h) *
                              w_feature +
                          w];
          }
        }
      }
    }
  }
}

void psamask_distribute_forward(const int num_, const int h_feature,
                                const int w_feature, const int h_mask,
                                const int w_mask, const int half_h_mask,
                                const int half_w_mask, const float *mask_data,
                                float *buffer_data) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            buffer_data[(n * h_feature * w_feature + h * w_feature + w) *
                            h_feature * w_feature +
                        (hidx + h - half_h_mask) * w_feature +
                        (widx + w - half_w_mask)] =
                mask_data[((n * h_mask * w_mask + hidx * w_mask + widx) *
                               h_feature +
                           h) *
                              w_feature +
                          w];
          }
        }
      }
    }
  }
}

void psamask_collect_backward(const int num_, const int h_feature,
                              const int w_feature, const int h_mask,
                              const int w_mask, const int half_h_mask,
                              const int half_w_mask, const float *buffer_diff,
                              float *mask_diff) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            mask_diff[((n * h_mask * w_mask + hidx * w_mask + widx) *
                           h_feature +
                       h) *
                          w_feature +
                      w] = buffer_diff[(n * h_feature * w_feature +
                                        (hidx + h - half_h_mask) * w_feature +
                                        (widx + w - half_w_mask)) *
                                           h_feature * w_feature +
                                       h * w_feature + w];
          }
        }
      }
    }
  }
}

void psamask_distribute_backward(const int num_, const int h_feature,
                                 const int w_feature, const int h_mask,
                                 const int w_mask, const int half_h_mask,
                                 const int half_w_mask,
                                 const float *buffer_diff, float *mask_diff) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            mask_diff[((n * h_mask * w_mask + hidx * w_mask + widx) *
                           h_feature +
                       h) *
                          w_feature +
                      w] =
                buffer_diff[(n * h_feature * w_feature + h * w_feature + w) *
                                h_feature * w_feature +
                            (hidx + h - half_h_mask) * w_feature +
                            (widx + w - half_w_mask)];
          }
        }
      }
    }
  }
}

void psamask_forward_cpu(HostContext &ctx, const SSElement &attr,
                         const OperatorBase::in_list_t &ins,
                         OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();
  const auto &input = ins[0];
  auto &output = outs[0];

  auto input_ptr = input.ptr<float>();
  auto output_ptr = output.ptr<float>();

  if (psa_type == 0)
    psamask_collect_forward(num_, h_feature, w_feature, h_mask, w_mask,
                            half_h_mask, half_w_mask, input_ptr, output_ptr);
  else
    psamask_distribute_forward(num_, h_feature, w_feature, h_mask, w_mask,
                               half_h_mask, half_w_mask, input_ptr, output_ptr);
}

void psamask_backward_cpu(HostContext &ctx, const SSElement &attr,
                          const OperatorBase::in_list_t &ins,
                          OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();

  const auto &input = ins[0];
  auto &output = outs[0];

  auto input_ptr = input.ptr<float>();
  auto output_ptr = output.ptr<float>();

  if (psa_type == 0)
    psamask_collect_backward(num_, h_feature, w_feature, h_mask, w_mask,
                             half_h_mask, half_w_mask, input_ptr, output_ptr);
  else
    psamask_distribute_backward(num_, h_feature, w_feature, h_mask, w_mask,
                                half_h_mask, half_w_mask, input_ptr,
                                output_ptr);
}

void PSAMaskForwardCUDAKernelLauncher(const int psa_type,
                                      const DArrayLite input, DArrayLite output,
                                      const int num_, const int h_feature,
                                      const int w_feature, const int h_mask,
                                      const int w_mask, const int half_h_mask,
                                      const int half_w_mask, CudaContext &ctx);

void PSAMaskBackwardCUDAKernelLauncher(const int psa_type,
                                       const DArrayLite grad_output,
                                       DArrayLite grad_input, const int num_,
                                       const int h_feature, const int w_feature,
                                       const int h_mask, const int w_mask,
                                       const int half_h_mask,
                                       const int half_w_mask, CudaContext &ctx);

void psamask_forward_cuda(CudaContext &ctx, const SSElement &attr,
                          const OperatorBase::in_list_t &ins,
                          OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();
  const auto &input = ins[0];
  auto &output = outs[0];
  PSAMaskForwardCUDAKernelLauncher(psa_type, input, output, num_, h_feature,
                                   w_feature, h_mask, w_mask, half_h_mask,
                                   half_w_mask, ctx);
}

void psamask_backward_cuda(CudaContext &ctx, const SSElement &attr,
                           const OperatorBase::in_list_t &ins,
                           OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();

  const auto &input = ins[0];
  auto &output = outs[0];
  PSAMaskBackwardCUDAKernelLauncher(psa_type, input, output, num_, h_feature,
                                    w_feature, h_mask, w_mask, half_h_mask,
                                    half_w_mask, ctx);
}

PARROTS_EXTENSION_REGISTER(psamask_forward)
    .attr("psa_type")
    .attr("num_")
    .attr("h_feature")
    .attr("w_feature")
    .attr("h_mask")
    .attr("w_mask")
    .attr("half_h_mask")
    .attr("half_w_mask")
    .input(1)
    .output(1)
    .apply(psamask_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(psamask_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(psamask_backward)
    .attr("psa_type")
    .attr("num_")
    .attr("h_feature")
    .attr("w_feature")
    .attr("h_mask")
    .attr("w_mask")
    .attr("half_h_mask")
    .attr("half_w_mask")
    .input(1)
    .output(1)
    .apply(psamask_backward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(psamask_backward_cuda)
#endif
    .done();
