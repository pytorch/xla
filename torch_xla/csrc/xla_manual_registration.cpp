#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_xla/csrc/aten_cpu_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/ops/nms.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace manual {
namespace {

struct NmsOp {
  using schema = at::Tensor(const at::Tensor&, const at::Tensor&, double);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchvision::nms")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
};

at::Tensor nms_kernel(const at::Tensor& boxes, const at::Tensor& scores,
                      double iou_threshold) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  if (!DebugUtil::ExperimentEnabled("nms")) {
    return at::native::call_fallback_fn<&xla_fallback, NmsOp>::call(
        boxes, scores, iou_threshold);
  }

  XLA_CHECK_EQ(boxes.dim(), 2) << "nms(): boxes should be a 2D tensor.";
  XLA_CHECK_EQ(boxes.size(1), 4)
      << "nms(): boxes should be a 2D tensor of shape [N, 4].";
  XLA_CHECK_EQ(scores.dim(), 1) << "nms(): scores should be a 1D tensor.";
  XLA_CHECK_EQ(boxes.size(0), scores.size(0))
      << "nms(): boxes and scores should have the same size for dimension 0.";

  XLATensorPtr xla_boxes = bridge::GetXlaTensor(boxes);
  XLATensorPtr xla_scores = bridge::GetXlaTensor(scores);
  return bridge::AtenFromXlaTensor(
      tensor_methods::nms(xla_boxes, xla_scores, iou_threshold),
      /*skip_functionalization=*/true);
}

}  // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

}  // namespace manual
}  // namespace torch_xla
