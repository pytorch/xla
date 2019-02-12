#include "ops/conv2d_backward.h"

#include "absl/strings/str_join.h"
#include "convolution.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& grad_output, const Value& input, const Value& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [stride, padding](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3)
        << "Unexpected number of operands: " << operands.size();
    // The precision doesn't matter for shape inference.
    Conv2DGrads grads =
        BuildConv2dBackward(operands[0], operands[1], operands[2], stride,
                            padding, xla::PrecisionConfig::DEFAULT);
    return xla::Tuple(operands[0].builder(),
                      {grads.grad_input, grads.grad_weight, grads.grad_bias});
  };
  return InferOutputShape(
      {grad_output->shape(), input->shape(), weight->shape()},
      lower_for_shape_fn);
}

xla::PrecisionConfig::Precision MakePrecisionConfig(
    bool use_full_conv_precision) {
  return use_full_conv_precision ? xla::PrecisionConfig::HIGHEST
                                 : xla::PrecisionConfig::DEFAULT;
}

}  // namespace

Conv2dBackward::Conv2dBackward(
    const Value& grad_output, const Value& input, const Value& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool use_full_conv_precision)
    : Node(ir::OpKind(at::aten::thnn_conv2d_backward),
           {grad_output, input, weight},
           NodeOutputShape(grad_output, input, weight, stride, padding),
           /*num_outputs=*/3, xla::util::MHash(stride, padding)),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      precision_(MakePrecisionConfig(use_full_conv_precision)) {}

XlaOpVector Conv2dBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  auto grads = BuildConv2dBackward(grad_output, input, weight, stride_,
                                   padding_, precision_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string Conv2dBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", stride=[" << absl::StrJoin(stride_, ", ")
     << "], padding=[" << absl::StrJoin(padding_, ", ")
     << "], precision=" << xla::PrecisionConfig::Precision_Name(precision_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
