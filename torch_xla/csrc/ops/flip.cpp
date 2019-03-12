#include "torch_xla/csrc/ops/flip.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Flip::Flip(const Value& input, std::vector<xla::int64> dims)
    : Node(ir::OpKind(at::aten::flip), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dims)),
      dims_(std::move(dims)) {}

XlaOpVector Flip::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Rev(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Flip::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=[" << absl::StrJoin(dims_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
