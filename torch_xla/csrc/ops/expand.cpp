#include "torch_xla/csrc/ops/expand.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const std::vector<int64_t>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], size);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Expand::Expand(const Value& input, std::vector<int64_t> size)
    : Node(ir::OpKind(at::aten::expand), {input},
           [&]() { return NodeOutputShape(input, size); },
           /*num_outputs=*/1, torch::lazy::MHash(size)),
      size_(std::move(size)) {}

NodePtr Expand::Clone(OpList operands) const {
  return MakeNode<Expand>(operands.at(0), size_);
}

XlaOpVector Expand::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildExpand(input, size_), loctx);
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
