#include "torch_xla/csrc/cross_replica_reduces.h"

#include <map>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/layout_manager.h"

namespace torch_xla {
namespace {

struct PerTypeContext {
  std::vector<xla::XlaOp> ops;
  std::vector<size_t> indices;
  std::vector<xla::Shape> operand_shapes;
};

struct ReduceContext {
  std::map<xla::PrimitiveType, PerTypeContext> contexts;
};

xla::Shape MakeReduceShape(absl::Span<const xla::Shape> operand_shapes) {
  Device xla_device = GetCurrentDevice();
  std::vector<xla::Shape> shapes_and_layouts;
  shapes_and_layouts.reserve(operand_shapes.size());
  for (auto& shape : operand_shapes) {
    shapes_and_layouts.push_back(MakeArrayShapeFromDimensions(
        shape.dimensions(), shape.dynamic_dimensions(), shape.element_type(),
        xla_device.hw_type));
  }
  return xla::ShapeUtil::MakeTupleShape(shapes_and_layouts);
}

ReduceContext GetReduceContext(absl::Span<const xla::XlaOp> operands) {
  ReduceContext redux;
  for (size_t i = 0; i < operands.size(); ++i) {
    xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operands[i]);
    PerTypeContext& ctx = redux.contexts[operand_shape.element_type()];
    ctx.ops.push_back(operands[i]);
    ctx.indices.push_back(i);
    ctx.operand_shapes.push_back(std::move(operand_shape));
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

std::vector<xla::ReplicaGroup> CreateReduceGroups(
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> reduce_groups;
  for (auto& group : groups) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    reduce_groups.push_back(std::move(rgroup));
  }
  return reduce_groups;
}

xla::XlaOp SliceOneToken(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (input_shape.rank() == 0) {
    return input;
  }
  return xla::SliceInDim(input, 0, 1, 1, 0);
}

}  // namespace

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> operands,
    xla::XlaOp token, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> reduce_groups = CreateReduceGroups(groups);
  // TODO: We use pseudo-tokens ATM, which are real values. This need to be
  // switched to use the real XLA Token once support has been added to XLA
  // AllReduce().
  xla::XlaOp chained_token = token;
  ReduceContext redux = GetReduceContext(operands);
  std::vector<xla::XlaOp> result(operands.size());
  for (auto& type_ctx : redux.contexts) {
    xla::XlaOp token_op = MaybeConvertTo(chained_token, type_ctx.first);
    type_ctx.second.ops.push_back(token_op);
    type_ctx.second.operand_shapes.push_back(
        XlaHelpers::ShapeOfXlaOp(token_op));

    xla::XlaOp reduce = xla::AllReduce(
        xla::Tuple(operands[0].builder(), type_ctx.second.ops),
        GetReduceComutation(reduce_type, type_ctx.first), reduce_groups,
        /*channel_id=*/absl::nullopt,
        MakeReduceShape(type_ctx.second.operand_shapes));
    for (size_t i = 0; i < type_ctx.second.indices.size(); ++i) {
      size_t op_idx = type_ctx.second.indices[i];
      xla::XlaOp gte = xla::GetTupleElement(reduce, i);
      if (scale != 1.0) {
        xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
            scale, type_ctx.second.operand_shapes[i].element_type(),
            gte.builder());
        gte = gte * scaling_value;
      }
      result[op_idx] = gte;
    }
    chained_token =
        xla::GetTupleElement(reduce, type_ctx.second.indices.size());
  }
  result.push_back(
      MaybeConvertTo(chained_token, XlaHelpers::TypeOfXlaOp(token)));
  return result;
}

AllToAllResult BuildAllToAll(
    xla::XlaOp input, xla::XlaOp token, xla::int64 split_dimension,
    xla::int64 concat_dimension, xla::int64 split_count,
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> reduce_groups = CreateReduceGroups(groups);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp affine_token = MaybeConvertTo(token, input_shape.element_type());
  xla::Shape reduce_shape = MakeArrayShapeFromDimensions(
      input_shape.dimensions(), input_shape.dynamic_dimensions(),
      input_shape.element_type(), GetCurrentDevice().hw_type);
  xla::XlaOp reduce_result =
      xla::AllToAll(input + affine_token, split_dimension, concat_dimension,
                    split_count, reduce_groups, reduce_shape.layout());
  xla::XlaOp chained_token =
      MaybeConvertTo(affine_token * SliceOneToken(reduce_result),
                     XlaHelpers::TypeOfXlaOp(token));
  return {reduce_result, chained_token};
}

CollectivePermuteResult BuildCollectivePermute(
    xla::XlaOp input, xla::XlaOp token,
    const std::vector<std::pair<xla::int64, xla::int64>>& source_target_pairs) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp affine_token = MaybeConvertTo(token, input_shape.element_type());
  // TODO: This is missing layout pinning ATM. If XLA scheduling is not exactly
  // the same (graphs on cores differ), XLA could assign different layouts and
  // things will break.
  xla::XlaOp result =
      xla::CollectivePermute(input + affine_token, source_target_pairs);
  xla::XlaOp chained_token = MaybeConvertTo(
      affine_token * SliceOneToken(result), XlaHelpers::TypeOfXlaOp(token));
  return {result, chained_token};
}

}  // namespace torch_xla
