#include "ops/max_pool2d_backward.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "pooling.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& grad_output, const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [stride, padding,
       kernel_size](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildMaxPool2dBackward(/*out_backprop=*/operands[0],
                                  /*input=*/operands[1], kernel_size, stride,
                                  padding);
  };
  return InferOutputShape({grad_output.node->shape(), input.node->shape()},
                          lower_for_shape_fn);
}

}  // namespace

MaxPool2dBackward::MaxPool2dBackward(
    const Value& grad_output, const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding)
    : Node(ir::OpKind(at::aten::max_pool2d_with_indices_backward),
           {grad_output, input},
           NodeOutputShape(grad_output, input, kernel_size, stride, padding),
           /*num_outputs=*/1, xla::util::MHash(kernel_size, stride, padding)),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()) {}

XlaOpVector MaxPool2dBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildMaxPool2dBackward(
      /*out_backprop=*/grad_output, /*input=*/input, kernel_size_, stride_,
      padding_);
  return ReturnOp(output, loctx);
}

std::string MaxPool2dBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", kernel_size=["
     << absl::StrJoin(kernel_size_, ", ") << "], stride=["
     << absl::StrJoin(stride_, ", ") << "], padding=["
     << absl::StrJoin(padding_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
