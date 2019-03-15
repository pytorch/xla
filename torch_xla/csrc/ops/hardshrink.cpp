#include "torch_xla/csrc/ops/hardshrink.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

Hardshrink::Hardshrink(const Value& input, at::Scalar lambda)
    : Node(OpKind(at::aten::hardshrink), {input}, input.shape(),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Hardshrink::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

XlaOpVector Hardshrink::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildHardshrink(input, lambda_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
