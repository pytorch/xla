#include "torch_xla/csrc/ops/user_computation.h"

#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

size_t GetNumOutputs(const xla::Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
}

}  // namespace

UserComputation::UserComputation(torch::lazy::OpKind op,
                                 torch::lazy::OpList operands,
                                 ComputationPtr computation)
    : XlaNode(std::move(op), operands, computation->program_shape().result(),
              GetNumOutputs(computation->program_shape().result()),
              computation->hash()),
      computation_(std::move(computation)) {}

torch::lazy::NodePtr UserComputation::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UserComputation>(op(), operands, computation_);
}

XlaOpVector UserComputation::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (auto& op : operands()) {
    inputs.push_back(loctx->GetOutputOp(op));
  }
  xla::XlaOp output =
      xla::Call(loctx->builder(), computation_->computation(), inputs);
  XlaOpVector results;
  const xla::Shape& result_shape = computation_->program_shape().result();
  if (result_shape.IsTuple()) {
    for (int64_t i = 0; i < result_shape.tuple_shapes_size(); ++i) {
      results.push_back(xla::GetTupleElement(output, i));
    }
  } else {
    results.push_back(output);
  }
  return ReturnOps(results, loctx);
}

std::string UserComputation::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", computation=" << computation_->name();
  return ss.str();
}

}  // namespace torch_xla
