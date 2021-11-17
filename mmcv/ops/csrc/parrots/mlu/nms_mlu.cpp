#include <parrots/compute/aten.hpp>
#include <parrots_mlu_helper.hpp>

using namespace parrots;

template <typename T>
void nms_parrots(T& ctx, const SSElement& attr,
                 const OperatorBase::in_list_t& ins,
                 OperatorBase::out_list_t& outs) {}

#define USE_CPU_NMS

#ifdef USE_CPU_NMS

using at::Tensor;
Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset);

template <>
void nms_parrots<HostContext>(HostContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  CAMB_BENCHMARK_OP();
  float iou_threshold;
  int offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("offset", offset)
      .done();

  at::Tensor boxes, scores;
  boxes = buildATensor(ctx, ins[0]);
  scores = buildATensor(ctx, ins[1]);
  auto out = nms_cpu(boxes, scores, iou_threshold, offset);
  updateDArray(ctx, out, outs[0]);
  return;
}

#endif  // USE_CPU_NMS

#ifdef PARROTS_USE_CAMB

void KernelNms(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
               const cnrtDataType_t data_type_input, const void* boxes_ptr,
               const void* scores_ptr, const int input_num_boxes,
               const int input_stride, const int max_output_boxes,
               const float iou_threshold, const float offset,
               void* workspace_ptr, void* output_size_ptr, void* output_ptr);

void NMSMLUKernelLauncher(CambContext& ctx, const DArrayLite& boxes,
                          const DArrayLite& scores, DArrayLite& output,
                          const float iou_threshold, const int offset) {
  if (boxes.size() == 0) {
    output = ctx.createDArrayLite(boxes.spec().withElemType(Prim::Int32));
    return;
  }
  // dimension parameters check
  PARROTS_CHECKARGS(boxes.ndims() == 2)
      << "boxes should be a 2d tensor, got " << boxes.ndims() << "D";
  PARROTS_CHECKARGS(boxes.dim(1) == 4)
      << "boxes should have 4 elements in dimension 1, got " << boxes.dim(1);
  PARROTS_CHECKARGS(scores.ndims() == 1)
      << "scores should be a 1d tensor, got " << scores.ndims() << "D";
  // data type check
  PARROTS_CHECKARGS(boxes.elemType() == Prim::Float32 ||
                    boxes.elemType() == Prim::Float16)
      << "data type of boxes should be Float or Half, got " << boxes.elemType();
  PARROTS_CHECKARGS(boxes.elemType() == scores.elemType())
      << "boxes should have the same type as scores";

  int input_num_boxes = boxes.dim(0);
  int input_stride = boxes.dim(1);
  int max_output_boxes = boxes.dim(0);
  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_x = core_dim;
  cnrtDim3_t k_dim = {dim_x, 1, 1};
  cnrtDataType_t data_type_input = getCnrtDataType(boxes.elemType());

  DArrayLite output_tmp =
      ctx.createDArrayLite(boxes.spec()
                               .withElemType(Prim::Int32)
                               .withShape(DArrayShape(max_output_boxes)));
  DArrayLite output_size = ctx.createDArrayLite(
      scores.spec().withElemType(Prim::Int32).withShape(DArrayShape(1)));

  // workspace
  size_t space_size = 0;
  if (boxes.elemType() == Prim::Float16) {
    space_size = input_num_boxes * sizeof(int16_t);
  } else {
    space_size = input_num_boxes * sizeof(float);
  }
  auto workspace = ctx.createDArrayLite(DArraySpec::bytes(space_size));

  // get compute queue
  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  switch (k_type) {
    default: {
      PARROTS_CHECKARGS(false) << "[nms_mlu]:Failed to choose kernel to launch";
    }
    case CNRT_FUNC_TYPE_BLOCK:
    case CNRT_FUNC_TYPE_UNION1: {
      KernelNms(k_dim, k_type, queue, data_type_input, boxes.data(),
                scores.data(), input_num_boxes, input_stride, max_output_boxes,
                iou_threshold, offset, workspace.data(), output_size.data(),
                output_tmp.data());
    }; break;
  }

  int output_num = 0;
  PARROTS_CALLCNRT(cnrtMemcpyAsync(&output_num, output_size.data(), sizeof(int),
                                   queue, cnrtMemcpyDevToHost));

  PARROTS_CALLCNRT(cnrtSyncQueue(queue));
  output = ctx.createDArrayLite(boxes.spec()
                                    .withElemType(Prim::Int32)
                                    .withShape(DArrayShape(output_num)));
  PARROTS_CALLCNRT(cnrtMemcpyAsync(output.data(), output_tmp.data(),
                                   output.nbytes(), queue, cnrtMemcpyDevToDev));
}

template <>
void nms_parrots<CambContext>(CambContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  float iou_threshold;
  int offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("offset", offset)
      .done();
  const auto& boxes = ins[0];
  const auto& scores = ins[1];
  auto& out = outs[0];
  NMSMLUKernelLauncher(ctx, boxes, scores, out, iou_threshold, offset);
  return;
}
#endif  //  PARROTS_USE_CAMB

PARROTS_EXTENSION_REGISTER(nms)
    .attr("iou_threshold")
    .attr("offset")
    .input(2)
    .output(1)
#ifdef USE_CPU_NMS
    .apply(nms_parrots<HostContext>)
#endif
#ifdef PARROTS_USE_CAMB
    .apply(nms_parrots<CambContext>)
#endif
    .done();
