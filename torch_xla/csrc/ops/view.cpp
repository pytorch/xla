#include "ops/view.h"
#include "data_ops.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {

namespace {

xla::Shape NodeOutputShape(
    const NodeOperand& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes) {
  auto lower_for_shape_fn =
      [&output_sizes](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildView(operands[0], output_sizes);
  };
  return InferOutputShape({input.node->shape()}, lower_for_shape_fn);
}

}  // namespace

View::View(const NodeOperand& input,
           tensorflow::gtl::ArraySlice<const xla::int64> output_size)
    : Node(ir::OpKind(at::aten::view), {input},
           NodeOutputShape(input, output_size)),
      output_size_(output_size.begin(), output_size.end()) {}

XlaOpVector View::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildView(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string View::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
