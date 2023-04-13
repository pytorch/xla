#include "torch_xla/csrc/ops/amp_update_scale.h"

#include "xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& growth_tracker,
                           const torch::lazy::Value& current_scale) {
  return xla::ShapeUtil::MakeTupleShape(
      {GetXlaShape(growth_tracker), GetXlaShape(current_scale)});
}

}  // namespace

AmpUpdateScale::AmpUpdateScale(const torch::lazy::Value& current_scale,
                               const torch::lazy::Value& growth_tracker,
                               const torch::lazy::Value& found_inf,
                               double scale_growth_factor,
                               double scale_backoff_factor, int growth_interval)
    : XlaNode(torch::lazy::OpKind(at::aten::_amp_update_scale_),
              {current_scale, growth_tracker, found_inf},
              NodeOutputShape(growth_tracker, current_scale),
              /*num_outputs=*/2),
      scale_growth_factor_(scale_growth_factor),
      scale_backoff_factor_(scale_backoff_factor),
      growth_interval_(growth_interval) {}

torch::lazy::NodePtr AmpUpdateScale::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AmpUpdateScale>(
      operands[0], operands[1], operands[2], scale_growth_factor_,
      scale_backoff_factor_, growth_interval_);
}

XlaOpVector AmpUpdateScale::Lower(LoweringContext* loctx) const {
  return ReturnOps(
      BuildAmpUpdateScale(loctx->GetOutputOp(operand(0)),
                          loctx->GetOutputOp(operand(1)),
                          loctx->GetOutputOp(operand(2)), scale_growth_factor_,
                          scale_backoff_factor_, growth_interval_),
      loctx);
}

}  // namespace torch_xla
