#include "torch_xla/csrc/ops/dropout.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Dropout::Dropout(const Value& input, double probability)
    : Node(ir::OpKind(at::aten::dropout), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(probability)),
      probability_(probability) {}

XlaOpVector Dropout::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildDropout(input, probability_), loctx);
}

std::string Dropout::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", probability=" << probability_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
