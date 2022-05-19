#include "torch_xla/csrc/ops/convolution_backward_overrideable.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/convolution.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    bool transposed, absl::Span<const int64_t> output_padding, int64_t groups) {
  auto lower_for_shape_fn =
      [stride, padding, dilation, transposed, output_padding,
       groups](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3);
    // The precision doesn't matter for shape inference.
    ConvGrads grads = BuildConvolutionBackwardOverrideable(
        operands[0], operands[1], operands[2], stride, padding, dilation,
        transposed, output_padding, groups);
    return xla::Tuple(operands[0].builder(),
                      {grads.grad_input, grads.grad_weight, grads.grad_bias});
  };
  return InferOutputShape(
      {GetXlaShape(grad_output), GetXlaShape(input), GetXlaShape(weight)},
      lower_for_shape_fn);
}

}  // namespace

ConvolutionBackwardOverrideable::ConvolutionBackwardOverrideable(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups)
    : XlaNode(torch::lazy::OpKind(at::aten::convolution_backward_overrideable),
              {grad_output, input, weight},
              [&]() {
                return NodeOutputShape(grad_output, input, weight, stride,
                                       padding, dilation, transposed,
                                       output_padding, groups);
              },
              /*num_outputs=*/3,
              torch::lazy::MHash(stride, padding, dilation, transposed,
                                 output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {}

torch::lazy::NodePtr ConvolutionBackwardOverrideable::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ConvolutionBackwardOverrideable>(
      operands.at(0), operands.at(1), operands.at(2), stride_, padding_,
      dilation_, transposed_, output_padding_, groups_);
}

XlaOpVector ConvolutionBackwardOverrideable::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  auto grads = BuildConvolutionBackwardOverrideable(
      grad_output, input, weight, stride_, padding_, dilation_, transposed_,
      output_padding_, groups_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string ConvolutionBackwardOverrideable::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", stride=(" << absl::StrJoin(stride_, ", ")
     << "), padding=(" << absl::StrJoin(padding_, ", ") << "), dilation=("
     << absl::StrJoin(dilation_, ", ") << "), transpose=" << transposed_
     << ", output_padding=(" << absl::StrJoin(output_padding_, ", ")
     << "), groups=" << groups_;
  return ss.str();
}

}  // namespace torch_xla
