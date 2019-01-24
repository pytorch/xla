#include "ops/avg_pool2d.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "pooling.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(
    const NodeOperand& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) {
  auto lower_for_shape_fn =
      [stride, padding, kernel_size, count_include_pad](
          tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPool2d(operands[0], kernel_size, stride, padding,
                          count_include_pad);
  };
  return InferOutputShape({input.node->shape()}, lower_for_shape_fn);
}

}  // namespace

AvgPool2d::AvgPool2d(const NodeOperand& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
                     tensorflow::gtl::ArraySlice<const xla::int64> stride,
                     tensorflow::gtl::ArraySlice<const xla::int64> padding,
                     bool count_include_pad)
    : Node(ir::OpKind(at::aten::avg_pool2d), {input},
           NodeOutputShape(input, kernel_size, stride, padding,
                           count_include_pad)),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      count_include_pad_(count_include_pad) {}

XlaOpVector AvgPool2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildAvgPool2d(input, kernel_size_, stride_, padding_,
                                     count_include_pad_);
  return ReturnOp(output, loctx);
}

std::string AvgPool2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", kernel_size=["
     << absl::StrJoin(kernel_size_, ", ") << "], stride=["
     << absl::StrJoin(stride_, ", ") << "], padding=["
     << absl::StrJoin(padding_, ", ")
     << "], count_include_pad=" << count_include_pad_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
