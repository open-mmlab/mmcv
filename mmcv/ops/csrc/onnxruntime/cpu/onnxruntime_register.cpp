#include "onnxruntime_register.h"

#include "corner_pool.h"
#include "grid_sample.h"
#include "nms.h"
#include "ort_mmcv_utils.h"
#include "roi_align.h"
#include "roi_align_rotated.h"
#include "soft_nms.h"

const char *c_MMCVOpDomain = "mmcv";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
MMCVRoiAlignCustomOp c_MMCVRoiAlignCustomOp;
MMCVRoIAlignRotatedCustomOp c_MMCVRoIAlignRotatedCustomOp;
GridSampleOp c_GridSampleOp;
MMCVCornerPoolCustomOp c_MMCVCornerPoolCustomOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_NmsOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoiAlignCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoIAlignRotatedCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVCornerPoolCustomOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
