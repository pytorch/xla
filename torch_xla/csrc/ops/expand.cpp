#include "torch_xla/csrc/ops/expand.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const std::vector<int64_t>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], size);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Expand::Expand(const torch::lazy::Value& input, std::vector<int64_t> size)
    : XlaNode(
          torch::lazy::OpKind(at::aten::expand), {input},
          [&]() { return NodeOutputShape(input, size); },
          /*num_outputs=*/1, torch::lazy::MHash(size)),
      size_(std::move(size)) {}

torch::lazy::NodePtr Expand::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Expand>(operands.at(0), size_);
}

XlaOpVector Expand::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildExpand(input, size_), loctx);
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
