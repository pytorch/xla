#include "ops/conv2d.h"
#include "convolution.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {

namespace {

// The bias doesn't matter for shape inference.
xla::Shape NodeOutputShape(const NodeOperand& input, const NodeOperand& weight,
                           int stride, int padding) {
  std::vector<xla::int64> stride_2d(2, stride);
  std::vector<xla::int64> padding_2d(2, padding);
  auto lower_for_shape_fn =
      [&stride_2d,
       &padding_2d](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK(operands.size() == 2 || operands.size() == 3)
        << "Unexpected number of operands: " << operands.size();
    // The precision doesn't matter for shape inference.
    return BuildConvolution(operands[0], operands[1], stride_2d, padding_2d,
                            xla::PrecisionConfig::DEFAULT);
  };
  return InferOutputShape({input.node->shape(), weight.node->shape()},
                          lower_for_shape_fn);
}

xla::PrecisionConfig::Precision MakePrecisionConfig(
    bool use_full_conv_precision) {
  return use_full_conv_precision ? xla::PrecisionConfig::HIGHEST
                                 : xla::PrecisionConfig::DEFAULT;
}

}  // namespace

Conv2d::Conv2d(const NodeOperand& input, const NodeOperand& weight,
               const NodeOperand& bias, int stride, int padding,
               bool use_full_conv_precision)
    : Node(ir::OpKind(at::aten::convolution), {input, weight, bias},
           NodeOutputShape(input, weight, stride, padding)),
      stride_(stride),
      padding_(padding),
      precision_(MakePrecisionConfig(use_full_conv_precision)) {}

Conv2d::Conv2d(const NodeOperand& input, const NodeOperand& weight, int stride,
               int padding, bool use_full_conv_precision)
    : Node(ir::OpKind(at::aten::convolution), {input, weight},
           NodeOutputShape(input, weight, stride, padding)),
      stride_(stride),
      padding_(padding),
      precision_(MakePrecisionConfig(use_full_conv_precision)) {}

XlaOpVector Conv2d::Lower(LoweringContext* loctx) const {
  std::vector<xla::int64> stride_2d(2, stride_);
  std::vector<xla::int64> padding_2d(2, padding_);
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp kernel = loctx->GetOutputOp(operand(1));
  xla::XlaOp output;
  if (operands().size() == 3) {
    xla::XlaOp bias = loctx->GetOutputOp(operand(2));
    output = BuildConvolutionBias(input, kernel, bias, stride_2d, padding_2d,
                                  precision_);
  } else {
    XLA_CHECK_EQ(operands().size(), 2);
    output = BuildConvolution(input, kernel, stride_2d, padding_2d, precision_);
  }
  return ReturnOp(output, loctx);
}

std::string Conv2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", stride=" << stride_ << ", padding=" << padding_
     << ", precision=" << xla::PrecisionConfig::Precision_Name(precision_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
