#include "torch_xla/csrc/ops/log_softmax_backward.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/softmax_builder.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& output,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [dim](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildLogSoftmaxGrad(/*grad_output=*/operands[0],
                               /*output=*/operands[1], dim);
  };
  return InferOutputShape({grad_output.shape(), output.shape()},
                          lower_for_shape_fn);
}

}  // namespace

LogSoftmaxBackward::LogSoftmaxBackward(const Value& grad_output,
                                       const Value& output, xla::int64 dim)
    : Node(ir::OpKind(at::aten::_log_softmax_backward_data),
           {grad_output, output}, NodeOutputShape(grad_output, output, dim),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector LogSoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildLogSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output, dim_);
  return ReturnOp(grad_input, loctx);
}

std::string LogSoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
