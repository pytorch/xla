#include "torch_xla/csrc/ops/conv2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convolution.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

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
    Conv2DGrads grads = BuildConv2dBackward(operands[0], operands[1],
                                            operands[2], stride, padding);
    return xla::Tuple(operands[0].builder(),
                      {grads.grad_input, grads.grad_weight, grads.grad_bias});
  };
  return InferOutputShape({grad_output.shape(), input.shape(), weight.shape()},
                          lower_for_shape_fn);
}

}  // namespace

Conv2dBackward::Conv2dBackward(const Value& grad_output, const Value& input,
                               const Value& weight,
                               std::vector<xla::int64> stride,
                               std::vector<xla::int64> padding)
    : Node(
          ir::OpKind(at::aten::thnn_conv2d_backward),
          {grad_output, input, weight},
          [&]() {
            return NodeOutputShape(grad_output, input, weight, stride, padding);
          },
          /*num_outputs=*/3, xla::util::MHash(stride, padding)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

NodePtr Conv2dBackward::Clone(OpList operands) const {
  return MakeNode<Conv2dBackward>(operands.at(0), operands.at(1),
                                  operands.at(2), stride_, padding_);
}

XlaOpVector Conv2dBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  auto grads =
      BuildConv2dBackward(grad_output, input, weight, stride_, padding_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string Conv2dBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", stride=[" << absl::StrJoin(stride_, ", ")
     << "], padding=[" << absl::StrJoin(padding_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
