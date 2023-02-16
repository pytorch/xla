#include "torch_xla/csrc/ops/repeat.h"

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> repeats) {
  auto lower_for_shape_fn =
      [repeats](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildRepeat(operands[0], repeats);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Repeat::Repeat(const torch::lazy::Value& input, std::vector<int64_t> repeats)
    : XlaNode(torch::lazy::OpKind(at::aten::repeat), {input},
              [&]() { return NodeOutputShape(input, repeats); },
              /*num_outputs=*/1, torch::lazy::MHash(repeats)),
      repeats_(std::move(repeats)) {}

torch::lazy::NodePtr Repeat::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Repeat>(operands.at(0), repeats_);
}

XlaOpVector Repeat::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildRepeat(input, repeats_);
  return ReturnOp(output, loctx);
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", repeats=(" << absl::StrJoin(repeats_, ", ")
     << ")";
  return ss.str();
}

}  // namespace torch_xla
