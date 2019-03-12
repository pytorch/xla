#include "torch_xla/csrc/ops/max_pool2d.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [stride, padding,
       kernel_size](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildMaxPool2d(operands[0], kernel_size, stride, padding);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

MaxPool2d::MaxPool2d(const Value& input, std::vector<xla::int64> kernel_size,
                     std::vector<xla::int64> stride,
                     std::vector<xla::int64> padding)
    : Node(ir::OpKind(at::aten::max_pool2d), {input},
           NodeOutputShape(input, kernel_size, stride, padding),
           /*num_outputs=*/1, xla::util::MHash(kernel_size, stride, padding)),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()) {}

XlaOpVector MaxPool2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildMaxPool2d(input, kernel_size_, stride_, padding_);
  return ReturnOp(output, loctx);
}

std::string MaxPool2d::ToString() const {
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
