#include "torch_xla/csrc/ops/hardtanh_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

HardtanhBackward::HardtanhBackward(const Value& grad_output, const Value& input,
                                   at::Scalar min_val, at::Scalar max_val)
    : Node(OpKind(at::aten::hardtanh_backward), {grad_output, input},
           grad_output.shape(), /*num_outputs=*/1,
           xla::util::MHash(ScalarHash(min_val), ScalarHash(max_val))),
      min_val_(min_val),
      max_val_(max_val) {}

std::string HardtanhBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", min_val=" << min_val_
     << ", max_val=" << max_val_;
  return ss.str();
}

XlaOpVector HardtanhBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildHardtanhBackward(grad_output, input, min_val_, max_val_);
  return ReturnOp(output, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
