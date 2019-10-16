#include "torch_xla/csrc/ops/mse_loss.h"

#include <ATen/core/Reduction.h>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& target,
                           xla::int64 reduction) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return BuildMseLoss(operands[0], operands[1],
                        MseLoss::GetXlaReductionMode(reduction));
  };
  return InferOutputShape({input.shape(), target.shape()}, lower_for_shape_fn);
}

}  // namespace

MseLoss::MseLoss(const Value& input, const Value& target, xla::int64 reduction)
    : Node(ir::OpKind(at::aten::mse_loss), {input, target},
           [&]() { return NodeOutputShape(input, target, reduction); },
           /*num_outputs=*/1, xla::util::MHash(reduction)),
      reduction_(reduction) {}

NodePtr MseLoss::Clone(OpList operands) const {
  return MakeNode<MseLoss>(operands.at(0), operands.at(1), reduction_);
}

XlaOpVector MseLoss::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp target = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildMseLoss(input, target, GetXlaReductionMode(reduction_)),
                  loctx);
}

std::string MseLoss::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", reduction=" << reduction_;
  return ss.str();
}

ReductionMode MseLoss::GetXlaReductionMode(xla::int64 reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return ReductionMode::kMean;
    case at::Reduction::None:
      return ReductionMode::kNone;
    case at::Reduction::Sum:
      return ReductionMode::kSum;
  }
  XLA_ERROR() << "Unknown reduction mode: " << reduction;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
