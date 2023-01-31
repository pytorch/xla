#include "torch_xla/csrc/ops/squeeze.h"

#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerSqueeze(xla::XlaOp input, int dim) {
  if (dim == -1) {
    return SqueezeAllTrivialDimensions(input);
  }
  XLA_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int dim) {
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return LowerSqueeze(operands[0], dim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Squeeze::Squeeze(const torch::lazy::Value& input, int dim)
    : XlaNode(torch::lazy::OpKind(at::aten::squeeze), {input},
              [&]() { return NodeOutputShape(input, dim); },
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr Squeeze::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Squeeze>(operands.at(0), dim_);
}

XlaOpVector Squeeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerSqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Squeeze::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
