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

xla::Shape NodeOutputShape(const OpList& inputs, const Value& found_inf) {
  std::vector<xla::Shape> output_shapes;
  output_shapes.reserve(inputs.size() + 1);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const xla::Shape& input_shape = inputs[i].shape();
    output_shapes.push_back(input_shape);
  }
  output_shapes.push_back(found_inf.shape());
  return xla::ShapeUtil::MakeTupleShape(output_shapes);
}

std::vector<Value> GetOperandList(absl::Span<const Value> operands,
                                  const Value& found_inf,
                                  const Value& inv_scale) {
  std::vector<Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(found_inf);
  operand_list.push_back(inv_scale);
  return operand_list;
}

}  // namespace

AmpForachNonFiniteCheckAndUnscale::AmpForachNonFiniteCheckAndUnscale(
    const OpList& inputs, const Value& found_inf, const Value& inv_scale)
    : Node(ir::OpKind(at::aten::_amp_foreach_non_finite_check_and_unscale_),
           GetOperandList(inputs, found_inf, inv_scale),
           NodeOutputShape(inputs, found_inf),
           /*num_outputs=*/inputs.size() + 1) {}

NodePtr AmpForachNonFiniteCheckAndUnscale::Clone(OpList operands) const {
  std::vector<Value> operand_list(operands.begin(), operands.end() - 2);
  size_t sz = operand_list.size();
  return MakeNode<AmpForachNonFiniteCheckAndUnscale>(operand_list, operands[sz],
                                                     operands[sz + 1]);
}

XlaOpVector AmpForachNonFiniteCheckAndUnscale::Lower(
    LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (size_t i = 0; i < operands().size() - 2; ++i) {
    inputs.push_back(loctx->GetOutputOp(operand(i)));
  }
  return ReturnOps(
      BuildAmpForeachNonFiniteCheckAndUnscale(
          inputs, loctx->GetOutputOp(operand(operands().size() - 2)),
          loctx->GetOutputOp(operand(operands().size() - 1))),
      loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
