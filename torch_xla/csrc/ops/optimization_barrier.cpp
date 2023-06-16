#include "torch_xla/csrc/ops/optimization_barrier.h"

#include "xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::OpList& inputs) {
  std::vector<xla::Shape> output_shapes;
  output_shapes.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    output_shapes.push_back(GetXlaShape(inputs[i]));
  }
  return xla::ShapeUtil::MakeTupleShape(output_shapes);
}

}  // namespace

OptimizationBarrier::OptimizationBarrier(const torch::lazy::OpList& inputs)
    : XlaNode(xla_optimization_barrier, inputs, NodeOutputShape(inputs),
              /*num_outputs=*/inputs.size()) {}

torch::lazy::NodePtr OptimizationBarrier::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<OptimizationBarrier>(operands);
}

XlaOpVector OptimizationBarrier::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (size_t i = 0; i < operands().size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand(i)));
  }
  xla::XlaOp tuple_input = xla::Tuple(inputs[0].builder(), inputs);
  xla::XlaOp tuple_output = xla::OptimizationBarrier(tuple_input);
  std::vector<xla::XlaOp> outputs;
  outputs.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    outputs.push_back(xla::GetTupleElement(tuple_output, i));
  }
  return ReturnOps({outputs}, loctx);
}

}  // namespace torch_xla
