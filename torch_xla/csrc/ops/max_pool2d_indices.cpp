#include "ops/max_pool2d_indices.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "pooling.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

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
  xla::Shape shape =
      InferOutputShape({input.node->shape()}, lower_for_shape_fn);
  shape.set_element_type(xla::PrimitiveType::S64);
  return shape;
}

}  // namespace

MaxPool2dIndices::MaxPool2dIndices(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding)
    : Node(ir::OpKind(at::aten::max_pool2d_with_indices), {input},
           NodeOutputShape(input, kernel_size, stride, padding),
           /*num_outputs=*/1, xla::util::MHash(kernel_size, stride, padding)),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()) {}

XlaOpVector MaxPool2dIndices::Lower(LoweringContext* loctx) const {
  XLA_ERROR() << "Not supported";
  return {};
}

std::string MaxPool2dIndices::ToString() const {
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
