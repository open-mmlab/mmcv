#include "parrots_cpp_helper.hpp"
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


DArrayLite NMSCUDALauncher(
    const DArrayLite boxes_sorted, const DArrayLite order, const DArrayLite areas,
    float iou_threshold, int offset, CudaContext& ctx, cudaStream_t stream);

void nms_cuda(
    CudaContext& ctx,
    const SSElement& attr,
    const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {

    float iou_threshold;
    int offset;
    SSAttrs(attr)
        .get<float>("iou_threshold", iou_threshold)
        .get<int>("offset", offset).done();

    const auto& boxes_sorted = ins[0];
    const auto& order = ins[1];
    const auto& areas = ins[2];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    outs[0] = NMSCUDALauncher(boxes_sorted, order, areas, iou_threshold, offset, ctx, stream);
}

void nms_cpu(
    HostContext& ctx,
    const SSElement& attr,
    const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {

    float iou_threshold;
    int offset;
    SSAttrs(attr)
        .get<float>("iou_threshold", iou_threshold)
        .get<int>("offset", offset).done();

    const auto& boxes = ins[0];
    const auto& order = ins[1];
    const auto& areas = ins[2];

    size_t nboxes = boxes.shape().dim(0);
    size_t boxes_dim = boxes.shape().dim(1);

    auto select = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, nboxes), getHostProxy());
    select.setZeros(syncStream());

    if(boxes.size() == 0) {
        outs[0] = select;
        return;
    }

    fill(ctx, select, *toScalar(1));

    auto select_ptr = select.ptr<int64_t>();
    auto boxes_ptr = boxes.ptr<float>();
    auto order_ptr = order.ptr<int64_t>();
    auto areas_ptr = areas.ptr<float>();

    for (int64_t _i=0; _i < nboxes; _i++) {
        if (select_ptr[_i] == 0) continue;
        auto i = order_ptr[_i];
        auto ix1 = boxes_ptr[i * boxes_dim];
        auto iy1 = boxes_ptr[i * boxes_dim + 1];
        auto ix2 = boxes_ptr[i * boxes_dim + 2];
        auto iy2 = boxes_ptr[i * boxes_dim + 3];
        auto iarea = areas_ptr[i];
        for (int64_t _j = _i + 1; _j < nboxes; _j++) {
            if (select_ptr[_j] == 0) continue;
            auto j = order_ptr[_j];
            auto xx1 = fmaxf(ix1, boxes_ptr[j * boxes_dim]);
            auto yy1 = fmaxf(iy1, boxes_ptr[j * boxes_dim + 1]);
            auto xx2 = fminf(ix2, boxes_ptr[j * boxes_dim + 2]);
            auto yy2 = fminf(iy2, boxes_ptr[j * boxes_dim + 3]);

            auto w = fmaxf(0.0, xx2 - xx1 + offset);
            auto h = fmaxf(0.0, yy2 - yy1 + offset);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas_ptr[j] - inter);
            if (ovr >= iou_threshold) select_ptr[_j] = 0;
        }
    }
    outs[0] = select;
}


PARROTS_EXTENSION_REGISTER(nms)
    .attr("iou_threshold")
    .attr("offset")
    .input(3)
    .output(1)
    .apply(nms_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(nms_cuda)
#endif
    .done();
