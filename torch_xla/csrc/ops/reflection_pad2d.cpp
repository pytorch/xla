#include "torch_xla/csrc/ops/reflection_pad2d.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReflectionPad2d(operands[0], padding);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

ReflectionPad2d::ReflectionPad2d(const torch::lazy::Value& input,
                                 std::vector<int64_t> padding)
    : XlaNode(torch::lazy::OpKind(at::aten::reflection_pad2d), {input},
              [&]() { return NodeOutputShape(input, padding); },
              /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {}

torch::lazy::NodePtr ReflectionPad2d::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReflectionPad2d>(operands.at(0), padding_);
}

XlaOpVector ReflectionPad2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildReflectionPad2d(input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReflectionPad2d::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace torch_xla
