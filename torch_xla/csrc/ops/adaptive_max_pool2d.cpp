#include "torch_xla/csrc/ops/adaptive_max_pool2d.h"

#include "xla/xla_client/debug_macros.h"
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
    MaxPoolResult result = BuildAdaptiveMaxPoolNd(operands[0], output_size, 2);
    return xla::Tuple(operands[0].builder(), {result.result, result.indices});
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

AdaptiveMaxPool2d::AdaptiveMaxPool2d(const torch::lazy::Value& input,
                                     std::vector<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::adaptive_max_pool2d), {input},
              [&]() { return NodeOutputShape(input, output_size); },
              /*num_outputs=*/2, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

torch::lazy::NodePtr AdaptiveMaxPool2d::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AdaptiveMaxPool2d>(operands.at(0), output_size_);
}

XlaOpVector AdaptiveMaxPool2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  MaxPoolResult result = BuildAdaptiveMaxPoolNd(input, output_size_, 2);
  return ReturnOps({result.result, result.indices}, loctx);
}

std::string AdaptiveMaxPool2d::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
