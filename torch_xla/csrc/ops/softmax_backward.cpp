#include "torch_xla/csrc/ops/softmax_backward.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/softmax_builder.h"

namespace torch_xla {
namespace ir {
namespace ops {

SoftmaxBackward::SoftmaxBackward(const Value& grad_output, const Value& output,
                                 xla::int64 dim)
    : Node(ir::OpKind(at::aten::_softmax_backward_data), {grad_output, output},
           grad_output.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector SoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output, dim_);
  return ReturnOp(grad_input, loctx);
}

std::string SoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
