#include "torch_xla/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const OpList& inputs) {
  std::vector<xla::Shape> output_shapes;
  output_shapes.reserve(inputs.size() - 1);
  for (size_t i = 0; i < inputs.size() - 2; ++i) {
    const xla::Shape& input_shape = inputs[i].shape();
    output_shapes.push_back(input_shape);
  }
  output_shapes.push_back(xla::ShapeUtil::MakeShape(
      inputs[inputs.size() - 2].shape().element_type(), {}));
  return xla::ShapeUtil::MakeTupleShape(output_shapes);
}

}  // namespace

AmpForachNonFiniteCheckAndUnscale::AmpForachNonFiniteCheckAndUnscale(
    const OpList& inputs)
    : Node(xla_amp_foreach_non_finite_check_and_unscale, inputs,
           NodeOutputShape(inputs),
           /*num_outputs=*/inputs.size() - 1) {}

NodePtr AmpForachNonFiniteCheckAndUnscale::Clone(OpList operands) const {
  return MakeNode<AmpForachNonFiniteCheckAndUnscale>(operands);
}

XlaOpVector AmpForachNonFiniteCheckAndUnscale::Lower(
    LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (size_t i = 0; i < num_outputs() + 1; ++i) {
    inputs.push_back(loctx->GetOutputOp(operand(i)));
  }
  return ReturnOps(BuildAmpForachNonFiniteCheckAndUnscale(inputs), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla