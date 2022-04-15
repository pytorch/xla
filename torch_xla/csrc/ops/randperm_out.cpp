#include "torch_xla/csrc/ops/randperm_out.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(int64_t n) {
  auto lower_for_shape_fn_randperm_out =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildRandpermOut(n);
  };
  return InferOutputShape({}, lower_for_shape_fn_randperm_out);
}

}  // namespace

RandpermOut::RandpermOut(int64_t n)
    : Node(
          torch::lazy::OpKind(at::aten::randperm), {},
          [&]() { return NodeOutputShape(n); },
          /*num_outputs=*/1, torch::lazy::MHash(n)),
      n_(n) {}

torch::lazy::NodePtr RandpermOut::Clone(OpList operands) const {
  return MakeNode<RandpermOut>(n_);
}

XlaOpVector RandpermOut::Lower(LoweringContext* loctx) const {
  // xla::XlaOp out = loctx->GetOutputOp(operand(0));
  xla::XlaOp op_randperm_out = BuildRandpermOut(n_);
  return ReturnOp(op_randperm_out, loctx);
}

std::string RandpermOut::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", n=" << n_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla