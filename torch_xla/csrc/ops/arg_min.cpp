#include "torch_xla/csrc/ops/arg_min.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int64_t dim,
                           bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMin(operands[0], dim, keepdim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

ArgMin::ArgMin(const torch::lazy::Value& input, int64_t dim, bool keepdim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::argmin), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/1, torch::lazy::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

torch::lazy::NodePtr ArgMin::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ArgMin>(operands.at(0), dim_, keepdim_);
}

XlaOpVector ArgMin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildArgMin(input, dim_, keepdim_), loctx);
}

std::string ArgMin::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace torch_xla
