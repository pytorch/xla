#include "torch_xla/csrc/ops/adaptive_avg_pool3d.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> output_size) {
  auto lower_for_shape_fn =
      [output_size](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool3d(operands[0], output_size);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

AdaptiveAvgPool3d::AdaptiveAvgPool3d(const torch::lazy::Value& input,
                                     std::vector<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::adaptive_avg_pool3d), {input},
              [&]() { return NodeOutputShape(input, output_size); },
              /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

torch::lazy::NodePtr AdaptiveAvgPool3d::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AdaptiveAvgPool3d>(operands.at(0), output_size_);
}

XlaOpVector AdaptiveAvgPool3d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildAdaptiveAvgPool3d(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string AdaptiveAvgPool3d::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
