#include "torch_xla/csrc/ops/convolution_overrideable.h"

#include "absl/strings/str_join.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/convolution.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

// The bias doesn't matter for shape inference.
xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& weight,
                           absl::Span<const int64_t> stride,
                           absl::Span<const int64_t> padding,
                           absl::Span<const int64_t> dilation, bool transposed,
                           absl::Span<const int64_t> output_padding,
                           int64_t groups) {
  auto lower_for_shape_fn =
      [stride, padding, dilation, output_padding, transposed,
       groups](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK(operands.size() == 2 || operands.size() == 3);
    return BuildConvolutionOverrideable(operands[0], operands[1], stride,
                                        padding, dilation, transposed,
                                        output_padding, groups);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(weight)},
                          lower_for_shape_fn);
}

}  // namespace

ConvolutionOverrideable::ConvolutionOverrideable(
    const torch::lazy::Value& input, const torch::lazy::Value& weight,
    const torch::lazy::Value& bias, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups)
    : XlaNode(
          torch::lazy::OpKind(at::aten::convolution_overrideable),
          {input, weight, bias},
          [&]() {
            return NodeOutputShape(input, weight, stride, padding, dilation,
                                   transposed, output_padding, groups);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(stride, padding, dilation, transposed,
                             output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {}

ConvolutionOverrideable::ConvolutionOverrideable(
    const torch::lazy::Value& input, const torch::lazy::Value& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups)
    : XlaNode(
          torch::lazy::OpKind(at::aten::convolution_overrideable),
          {input, weight},
          [&]() {
            return NodeOutputShape(input, weight, stride, padding, dilation,
                                   transposed, output_padding, groups);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(stride, padding, dilation, transposed,
                             output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {}

torch::lazy::NodePtr ConvolutionOverrideable::Clone(
    torch::lazy::OpList operands) const {
  return operands.size() == 3
             ? torch::lazy::MakeNode<ConvolutionOverrideable>(
                   operands.at(0), operands.at(1), operands.at(2), stride_,
                   padding_, dilation_, transposed_, output_padding_, groups_)
             : torch::lazy::MakeNode<ConvolutionOverrideable>(
                   operands.at(0), operands.at(1), stride_, padding_, dilation_,
                   transposed_, output_padding_, groups_);
}

XlaOpVector ConvolutionOverrideable::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp kernel = loctx->GetOutputOp(operand(1));
  xla::XlaOp output;
  if (operands().size() == 3) {
    xla::XlaOp bias = loctx->GetOutputOp(operand(2));
    output = BuildConvolutionOverrideableBias(input, kernel, bias, stride_,
                                              padding_, dilation_, transposed_,
                                              output_padding_, groups_);
  } else {
    XLA_CHECK_EQ(operands().size(), 2);
    output = BuildConvolutionOverrideable(input, kernel, stride_, padding_,
                                          dilation_, transposed_,
                                          output_padding_, groups_);
  }
  return ReturnOp(output, loctx);
}

std::string ConvolutionOverrideable::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", stride=(" << absl::StrJoin(stride_, ", ")
     << "), padding=(" << absl::StrJoin(padding_, ", ") << "), dilation=("
     << absl::StrJoin(dilation_, ", ") << "), transpose=" << transposed_
     << ", output_padding=(" << absl::StrJoin(output_padding_, ", ")
     << "), groups=" << groups_;
  return ss.str();
}

}  // namespace torch_xla
