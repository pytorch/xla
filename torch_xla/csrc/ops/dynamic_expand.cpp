#include "torch_xla/csrc/ops/dynamic_expand.h"


namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const Value& size) {
  xla::Shape shape = size.shape();
  auto lower_for_shape_fn =
      [shape](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildDynamicExpand(operands[0], operands[1], shape);
  };
  return InferOutputShape({input.shape(), shape}, lower_for_shape_fn);
}

}  // namespace

DynamicExpand2::DynamicExpand2(Value& lhs, Value& sz)
    : Node(ir::OpKind(c10::Symbol::prim("_dynamic_expand2")), {lhs, sz}, 
           [&]() { return NodeOutputShape(input, sz); },
           /*num_outputs=*/1, torch::lazy::MHash(sz.shape)), /*TODO: cast lazy shape to xla shape */
      ) {}

XlaOpVector Lower(LoweringContext* loctx) const {
    XLA_CHECK(operands().size() == 2);
    xla::XlaOp input = loctx->GetOutputOp(operand(0));
    xla::XlaOp size_ = loctx->GetOutputOp(operand(1)); // TODO: confirm with Nick if .input is needed
    xla::Shape shape_ = operand(1).shape(); //TODO: cast lazy::shape to xla::Shape (xla::ShapeUtil::CastShape ?) - confirm with Nick
    return ReturnOp(BuildDynamicExpand(input, size_, shape_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
