#include "torch_xla/csrc/cross_replica_reduces.h"

#include <map>

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

struct PerTypeContext {
  std::vector<xla::XlaOp> ops;
  std::vector<size_t> indices;
};

struct ReduceContext {
  std::map<xla::PrimitiveType, PerTypeContext> contexts;
  std::vector<xla::Shape> operand_shapes;
};

ReduceContext GetReduceContext(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) {
  ReduceContext redux;
  for (size_t i = 0; i < operands.size(); ++i) {
    redux.operand_shapes.push_back(XlaHelpers::ShapeOfXlaOp(operands[i]));
    PerTypeContext& ctx =
        redux.contexts[redux.operand_shapes.back().element_type()];
    ctx.ops.push_back(operands[i]);
    ctx.indices.push_back(i);
  }
  return redux;
}

}  // namespace

std::vector<xla::XlaOp> BuildCrossReplicaSum(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands,
    const xla::XlaOp& token, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> crs_groups;
  for (auto& group : groups) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    crs_groups.push_back(std::move(rgroup));
  }
  // TODO: Chain reduces with xla::Token when support will show up.
  xla::XlaOp chained_token = token;
  ReduceContext redux = GetReduceContext(operands);
  std::vector<xla::XlaOp> result(operands.size());
  for (auto& type_ctx : redux.contexts) {
    xla::XlaOp crs = xla::CrossReplicaSum(
        xla::Tuple(operands[0].builder(), type_ctx.second.ops), crs_groups);
    for (size_t i = 0; i < type_ctx.second.indices.size(); ++i) {
      size_t op_idx = type_ctx.second.indices[i];
      xla::XlaOp gte = xla::GetTupleElement(crs, i);
      if (scale != 1.0) {
        xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
            scale, redux.operand_shapes[op_idx].element_type(), gte.builder());
        gte = gte * scaling_value;
      }
      result[op_idx] = gte;
    }
  }
  result.push_back(chained_token);
  return result;
}

}  // namespace torch_xla
