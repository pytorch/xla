#include "torch_xla/csrc/ops/user_computation.h"

#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

size_t GetNumOutputs(const xla::Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
}

}  // namespace

UserComputation::UserComputation(OpKind op, OpList operands,
                                 ComputationPtr computation)
    : Node(std::move(op), operands, computation->program_shape().result(),
           GetNumOutputs(computation->program_shape().result()),
           computation->hash()),
      computation_(std::move(computation)) {}

NodePtr UserComputation::Clone(OpList operands) const {
  return MakeNode<UserComputation>(op(), operands, computation_);
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
    for (xla::int64 i = 0; i < result_shape.tuple_shapes_size(); ++i) {
      results.push_back(xla::GetTupleElement(output, i));
    }
  } else {
    results.push_back(output);
  }
  return ReturnOps(results, loctx);
}

std::string UserComputation::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", computation=" << computation_->name();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
