#include "torch_xla/csrc/cross_replica_reduces.h"

#include <map>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
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

xla::XlaComputation GetReduceComutation(AllReduceType reduce_type,
                                        xla::PrimitiveType type) {
  switch (reduce_type) {
    case AllReduceType::kSum:
      return XlaHelpers::CreateAddComputation(type);
    case AllReduceType::kMul:
      return XlaHelpers::CreateMulComputation(type);
    case AllReduceType::kAnd:
      return XlaHelpers::CreateAndComputation(type);
    case AllReduceType::kOr:
      return XlaHelpers::CreateOrComputation(type);
    case AllReduceType::kMin:
      return XlaHelpers::CreateMinComputation(type);
    case AllReduceType::kMax:
      return XlaHelpers::CreateMaxComputation(type);
  }
  XLA_ERROR() << "Invalid reduce type: "
              << xla::util::GetEnumValue(reduce_type);
}

}  // namespace

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type,
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands, xla::XlaOp token,
    double scale, const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> reduce_groups;
  for (auto& group : groups) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    reduce_groups.push_back(std::move(rgroup));
  }
  // TODO: We use pseudo-tokens ATM, which are real values. This need to be
  // switched to use the real XLA Token once support has been added to XLA
  // AllReduce().
  xla::XlaOp chained_token = token;
  ReduceContext redux = GetReduceContext(operands);
  std::vector<xla::XlaOp> result(operands.size());
  for (auto& type_ctx : redux.contexts) {
    type_ctx.second.ops.push_back(
        xla::ConvertElementType(chained_token, type_ctx.first));

    xla::XlaOp reduce = xla::AllReduce(
        xla::Tuple(operands[0].builder(), type_ctx.second.ops),
        GetReduceComutation(reduce_type, type_ctx.first), reduce_groups);
    for (size_t i = 0; i < type_ctx.second.indices.size(); ++i) {
      size_t op_idx = type_ctx.second.indices[i];
      xla::XlaOp gte = xla::GetTupleElement(reduce, i);
      if (scale != 1.0) {
        xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
            scale, redux.operand_shapes[op_idx].element_type(), gte.builder());
        gte = gte * scaling_value;
      }
      result[op_idx] = gte;
    }
    chained_token =
        xla::GetTupleElement(reduce, type_ctx.second.indices.size());
  }
  result.push_back(
      xla::ConvertElementType(chained_token, XlaHelpers::TypeOfXlaOp(token)));
  return result;
}

}  // namespace torch_xla
