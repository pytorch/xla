#include "ops/max_pool2d.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "pooling.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(const NodeOperand& input, int kernel_size,
                           int stride, int padding) {
  std::vector<xla::int64> stride_2d(2, stride);
  std::vector<xla::int64> padding_2d(2, padding);
  std::vector<xla::int64> kernel_size_2d(2, kernel_size);
  auto lower_for_shape_fn =
      [&stride_2d, &padding_2d,
       &kernel_size_2d](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildMaxPool2d(operands[0], kernel_size_2d, stride_2d, padding_2d);
  };
  return InferOutputShape({input.node->shape()}, lower_for_shape_fn);
}

}  // namespace

MaxPool2d::MaxPool2d(const NodeOperand& input, int kernel_size, int stride,
                     int padding)
    : Node(ir::OpKind(at::aten::max_pool2d), {input},
           NodeOutputShape(input, kernel_size, stride, padding)),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {}

XlaOpVector MaxPool2d::Lower(LoweringContext* loctx) const {
  std::vector<xla::int64> stride_2d(2, stride_);
  std::vector<xla::int64> padding_2d(2, padding_);
  std::vector<xla::int64> kernel_size_2d(2, kernel_size_);
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      BuildMaxPool2d(input, kernel_size_2d, stride_2d, padding_2d);
  return ReturnOp(output, loctx);
}

std::string MaxPool2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", kernel_size=" << kernel_size_
     << ", stride=" << stride_ << ", padding=" << padding_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
