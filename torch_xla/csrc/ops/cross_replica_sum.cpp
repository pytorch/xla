#include "torch_xla/csrc/ops/cross_replica_sum.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(tensorflow::gtl::ArraySlice<const Value> operands,
                           const Value& token) {
  std::vector<xla::Shape> tuple_shapes;
  tuple_shapes.reserve(operands.size() + 1);
  for (auto& operand : operands) {
    tuple_shapes.push_back(operand.shape());
  }
  tuple_shapes.push_back(token.shape());
  return xla::ShapeUtil::MakeTupleShape(tuple_shapes);
}

std::vector<Value> GetOperandList(
    tensorflow::gtl::ArraySlice<const Value> operands, const Value& token) {
  std::vector<Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(token);
  return operand_list;
}

}  // namespace

CrossReplicaSum::CrossReplicaSum(
    tensorflow::gtl::ArraySlice<const Value> operands, const Value& token,
    double scale, std::vector<std::vector<xla::int64>> groups)
    : Node(xla_cross_replica_sum, GetOperandList(operands, token),
           [&]() { return NodeOutputShape(operands, token); },
           /*num_outputs=*/operands.size() + 1,
           xla::util::MHash(scale, groups)),
      scale_(scale),
      groups_(std::move(groups)) {}

NodePtr CrossReplicaSum::Clone(OpList operands) const {
  std::vector<Value> operand_list(operands.begin(), operands.end() - 1);
  return MakeNode<CrossReplicaSum>(operand_list, operands.back(), scale_,
                                   groups_);
}

XlaOpVector CrossReplicaSum::Lower(LoweringContext* loctx) const {
  auto& operand_list = operands();
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operand_list.size());
  for (size_t i = 0; i + 1 < operand_list.size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand_list[i]));
  }
  xla::XlaOp token = loctx->GetOutputOp(operand_list.back());
  return ReturnOps(BuildCrossReplicaSum(inputs, token, scale_, groups_), loctx);
}

std::string CrossReplicaSum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", scale=" << scale_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
