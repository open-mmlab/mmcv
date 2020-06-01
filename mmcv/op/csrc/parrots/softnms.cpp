#include "parrots_cpp_helper.hpp"


void SoftNMSCPULauncher(
    HostContext& ctx,
    const SSElement& attr,
    const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {

    float iou_threshold;
    float sigma;
    float min_score;
    int method;
    int offset;
    SSAttrs(attr).get<float>("iou_threshold", iou_threshold)
                 .get<float>("sigma", sigma)
                 .get<float>("min_score", min_score)
                 .get<int>("method", method)
                 .get<int>("offset", offset)
                 .done();

    const auto& boxes = ins[0];
    const auto& scores = ins[1];
    const auto& areas = ins[2];

    size_t nboxes = boxes.shape().dim(0);
    size_t boxes_dim = boxes.shape().dim(1);
    auto boxes_ptr = boxes.ptr<float>();
    auto scores_ptr = scores.ptr<float>();
    auto areas_ptr = areas.ptr<float>();

    auto inputs = ctx.createDArrayLite(DArraySpec::array(Prim::Float32, DArrayShape(nboxes, 6)));
    auto inputs_ptr = inputs.ptr<float>();
    auto dets = ctx.createDArrayLite(DArraySpec::array(Prim::Float32, DArrayShape(nboxes, 5)));
    auto de = dets.ptr<float>();
    for (size_t i = 0; i < nboxes; i++) {
        inputs_ptr[i * 6 + 0] = boxes_ptr[i * boxes_dim + 0];
        inputs_ptr[i * 6 + 1] = boxes_ptr[i * boxes_dim + 1];
        inputs_ptr[i * 6 + 2] = boxes_ptr[i * boxes_dim + 2];
        inputs_ptr[i * 6 + 3] = boxes_ptr[i * boxes_dim + 3];
        inputs_ptr[i * 6 + 4] = scores_ptr[i];
        inputs_ptr[i * 6 + 5] = areas_ptr[i];
    }

    size_t pos = 0;
    auto inds_t = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, nboxes));
    arange(ctx, *toScalar(0), *toScalar(nboxes), *toScalar(1), inds_t);
    auto inds = inds_t.ptr<int64_t>();
    auto num_out = ctx.createDArrayLite(DArraySpec::scalar(Prim::Int64));


    for (size_t i = 0; i < nboxes; i++) {
        auto max_score = inputs_ptr[i * 6 + 4];
        auto max_pos = i;

        pos = i + 1;
        // get max box
        while (pos < nboxes) {
            if (max_score < inputs_ptr[pos * 6 + 4]) {
                max_score = inputs_ptr[pos * 6 + 4];
                max_pos = pos;
            }
            pos = pos + 1;
        }
        // swap
        auto ix1 = de[i * 5 + 0] = inputs_ptr[max_pos * 6 + 0];
        auto iy1 = de[i * 5 + 1] = inputs_ptr[max_pos * 6 + 1];
        auto ix2 = de[i * 5 + 2] = inputs_ptr[max_pos * 6 + 2];
        auto iy2 = de[i * 5 + 3] = inputs_ptr[max_pos * 6 + 3];
        auto iscore = de[i * 5 + 4] = inputs_ptr[max_pos * 6 + 4];
        auto iarea = inputs_ptr[max_pos * 6 + 5];
        auto iind = inds[max_pos];
        inputs_ptr[max_pos * 6 + 0] = inputs_ptr[i * 6 + 0];
        inputs_ptr[max_pos * 6 + 1] = inputs_ptr[i * 6 + 1];
        inputs_ptr[max_pos * 6 + 2] = inputs_ptr[i * 6 + 2];
        inputs_ptr[max_pos * 6 + 3] = inputs_ptr[i * 6 + 3];
        inputs_ptr[max_pos * 6 + 4] = inputs_ptr[i * 6 + 4];
        inputs_ptr[max_pos * 6 + 5] = inputs_ptr[i * 6 + 5];
        inds[max_pos] = inds[i];
        inputs_ptr[i * 6 + 0] = ix1;
        inputs_ptr[i * 6 + 1] = iy1;
        inputs_ptr[i * 6 + 2] = ix2;
        inputs_ptr[i * 6 + 3] = iy2;
        inputs_ptr[i * 6 + 4] = iscore;
        inputs_ptr[i * 6 + 5] = iarea;
        inds[i] = iind;

        pos = i + 1;
        while (pos < nboxes){
            auto xx1 = fmaxf(ix1, inputs_ptr[pos * 6 + 0]);
            auto yy1 = fmaxf(iy1, inputs_ptr[pos * 6 + 1]);
            auto xx2 = fminf(ix2, inputs_ptr[pos * 6 + 2]);
            auto yy2 = fminf(iy2, inputs_ptr[pos * 6 + 3]);

            auto w = fmaxf(0.0, xx2 - xx1 + offset);
            auto h = fmaxf(0.0, yy2 - yy1 + offset);
            auto inter = w * h;
            auto ovr = inter / (iarea + inputs_ptr[pos * 6 + 5] - inter);

            float weight = 1.;
            if (method == 0) {
                if (ovr >= iou_threshold) weight = 0;
            } else if (method == 1) {
                if (ovr >= iou_threshold) weight = 1 - ovr;
            } else if (method == 2) {
                weight = exp(-(ovr * ovr) / sigma);
            }
            inputs_ptr[pos * 6 + 4] *= weight;
            // if box score falls below threshold, discard the box by
            // swapping with last box update N
            if (inputs_ptr[pos * 6 + 4] < min_score) {
                inputs_ptr[pos * 6 + 0] = inputs_ptr[(nboxes - 1) * 6 + 0];
                inputs_ptr[pos * 6 + 1] = inputs_ptr[(nboxes - 1) * 6 + 1];
                inputs_ptr[pos * 6 + 2] = inputs_ptr[(nboxes - 1) * 6 + 2];
                inputs_ptr[pos * 6 + 3] = inputs_ptr[(nboxes - 1) * 6 + 3];
                inputs_ptr[pos * 6 + 4] = inputs_ptr[(nboxes - 1) * 6 + 4];
                inputs_ptr[pos * 6 + 5] = inputs_ptr[(nboxes - 1) * 6 + 5];
                inds[pos] = inds[nboxes - 1];
                nboxes = nboxes -1;
                pos = pos - 1;
            }
            pos = pos + 1;
        }
    }
    setScalar(num_out, int64_t{nboxes});
    outs[0] = dets;
    outs[1] = inds_t;
    outs[2] = num_out;
}


PARROTS_EXTENSION_REGISTER(softnms)
    .attr("iou_threshold")
    .attr("sigma")
    .attr("min_score")
    .attr("method")
    .attr("offset")
    .input(3)
    .output(3)
    .apply(SoftNMSCPULauncher)
    .done();
