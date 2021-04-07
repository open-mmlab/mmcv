#include "onnxruntime_register.h"

#include "grid_sample.h"
#include "nms.h"
#include "ort_mmcv_utils.h"
#include "roi_align.h"
#include "soft_nms.h"

const char *c_MMCVOpDomain = "mmcv";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
MMCVRoiAlignCustomOp c_MMCVRoiAlignCustomOp;
GridSampleOp c_GridSampleOp;

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

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
