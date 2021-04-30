#include "torch_xla/csrc/ops/amp_update_scale.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& growth_tracker,
                           const Value& current_scale) {
  return xla::ShapeUtil::MakeTupleShape(
      {growth_tracker.shape(), current_scale.shape()});
}

}  // namespace

AmpUpdateScale::AmpUpdateScale(const Value& current_scale,
                               const Value& growth_tracker,
                               const Value& found_inf,
                               double scale_growth_factor,
                               double scale_backoff_factor, int growth_interval)
    : Node(ir::OpKind(at::aten::_amp_update_scale_),
           {current_scale, growth_tracker, found_inf},
           NodeOutputShape(growth_tracker, current_scale),
           /*num_outputs=*/2),
      scale_growth_factor_(scale_growth_factor),
      scale_backoff_factor_(scale_backoff_factor),
      growth_interval_(growth_interval) {}

NodePtr AmpUpdateScale::Clone(OpList operands) const {
  return MakeNode<AmpUpdateScale>(operands[0], operands[1], operands[2],
                                  scale_growth_factor_, scale_backoff_factor_,
                                  growth_interval_);
}

XlaOpVector AmpUpdateScale::Lower(LoweringContext* loctx) const {
  return ReturnOps(
      BuildAmpUpdateScale(loctx->GetOutputOp(operand(0)),
                          loctx->GetOutputOp(operand(1)),
                          loctx->GetOutputOp(operand(2)), scale_growth_factor_,
                          scale_backoff_factor_, growth_interval_),
      loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
