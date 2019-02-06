#include "ops/softmax.h"
#include "log_softmax.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the log softmax operation.
xla::Shape NodeOutputShape(const Value& input, xla::int64 dim) {
  auto lower_for_shape_fn =
      [dim](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildLogSoftmax(operands[0], dim);
  };
  return InferOutputShape({input.node->shape()}, lower_for_shape_fn);
}

}  // namespace

LogSoftmax::LogSoftmax(const Value& input, xla::int64 dim)
    : Node(ir::OpKind(at::aten::log_softmax), {input},
           NodeOutputShape(input, dim), /*num_outputs=*/1,
           xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector LogSoftmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildLogSoftmax(input, dim_);
  return ReturnOp(output, loctx);
}

std::string LogSoftmax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
