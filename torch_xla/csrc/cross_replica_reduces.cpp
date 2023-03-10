#include "torch_xla/csrc/cross_replica_reduces.h"

#include <torch/csrc/lazy/core/util.h>

#include <map>

#include "tensorflow/compiler/xla/shape_util.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/token_handler.h"

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
  torch::lazy::BackendDevice xla_device = GetCurrentDevice();
  std::vector<xla::Shape> shapes_and_layouts;
  shapes_and_layouts.reserve(operand_shapes.size());
  for (auto& shape : operand_shapes) {
    shapes_and_layouts.push_back(MakeArrayShapeFromDimensions(
        shape.dimensions(), shape.dynamic_dimensions(), shape.element_type(),
        static_cast<XlaDeviceType>(xla_device.type())));
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
              << torch::lazy::GetEnumValue(reduce_type);
}

std::vector<xla::ReplicaGroup> CreateReduceGroups(
    const std::vector<std::vector<int64_t>>& groups) {
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

}  // namespace

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> operands,
    xla::XlaOp token, double scale,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout) {
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

    xla::XlaOp reduce;
    if (pin_layout) {
      reduce = xla::AllReduce(
          xla::Tuple(operands[0].builder(), type_ctx.second.ops),
          GetReduceComutation(reduce_type, type_ctx.first), reduce_groups,
          /*channel_id=*/absl::nullopt,
          /*shape_with_layout=*/
          MakeReduceShape(type_ctx.second.operand_shapes));
    } else {
      reduce = xla::AllReduce(
          xla::Tuple(operands[0].builder(), type_ctx.second.ops),
          GetReduceComutation(reduce_type, type_ctx.first), reduce_groups);
    }
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

AllToAllResult BuildAllToAll(xla::XlaOp input, xla::XlaOp token,
                             int64_t split_dimension, int64_t concat_dimension,
                             int64_t split_count,
                             const std::vector<std::vector<int64_t>>& groups,
                             bool pin_layout) {
  std::vector<xla::ReplicaGroup> reduce_groups = CreateReduceGroups(groups);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  TokenHandler token_handler(token);
  xla::XlaOp reduce_result;
  if (pin_layout) {
    torch::lazy::BackendDevice xla_device = GetCurrentDevice();
    xla::Shape reduce_shape = MakeArrayShapeFromDimensions(
        input_shape.dimensions(), input_shape.dynamic_dimensions(),
        input_shape.element_type(),
        static_cast<XlaDeviceType>(xla_device.type()));
    reduce_result = xla::AllToAll(token_handler.GetInput(input, &input_shape),
                                  split_dimension, concat_dimension,
                                  split_count, reduce_groups,
                                  /*layout=*/reduce_shape.layout());
  } else {
    reduce_result = xla::AllToAll(token_handler.GetInput(input, &input_shape),
                                  split_dimension, concat_dimension,
                                  split_count, reduce_groups);
  }
  return {reduce_result, token_handler.GetNewToken(reduce_result)};
}

AllGatherResult BuildAllGather(xla::XlaOp input, xla::XlaOp token, int64_t dim,
                               int64_t shard_count,
                               const std::vector<std::vector<int64_t>>& groups,
                               bool pin_layout) {
  std::vector<xla::ReplicaGroup> reduce_groups = CreateReduceGroups(groups);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  TokenHandler token_handler(token);
  xla::XlaOp all_gather_result;
  if (pin_layout) {
    torch::lazy::BackendDevice xla_device = GetCurrentDevice();
    xla::Shape reduce_shape = MakeArrayShapeFromDimensions(
        input_shape.dimensions(), input_shape.dynamic_dimensions(),
        input_shape.element_type(),
        static_cast<XlaDeviceType>(xla_device.type()));
    all_gather_result =
        xla::AllGather(token_handler.GetInput(input, &input_shape), dim,
                       shard_count, reduce_groups, /*channel_id=*/absl::nullopt,
                       /*layout=*/reduce_shape.layout());
  } else {
    all_gather_result =
        xla::AllGather(token_handler.GetInput(input, &input_shape), dim,
                       shard_count, reduce_groups);
  }
  return {all_gather_result, token_handler.GetNewToken(all_gather_result)};
}

CollectivePermuteResult BuildCollectivePermute(
    xla::XlaOp input, xla::XlaOp token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  TokenHandler token_handler(token);
  // TODO: This is missing layout pinning ATM. If XLA scheduling is not exactly
  // the same (graphs on cores differ), XLA could assign different layouts and
  // things will break.
  xla::XlaOp result = xla::CollectivePermute(
      token_handler.GetInput(input, &input_shape), source_target_pairs);
  return {result, token_handler.GetNewToken(result)};
}

SendResult BuildSendWithToken(xla::XlaOp input, xla::XlaOp token,
                              int64_t channel_id) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(channel_id);
  channel_handle.set_type(xla::ChannelHandle::DEVICE_TO_DEVICE);
  xla::XlaOp result_token = xla::SendWithToken(input, token, channel_handle);
  // Bind input into the result, so that the caller can depend on the result.
  // This can enable building the `send` op into the graph when the token
  // is ignored by some caller like `torch.distributed`.
  xla::XlaOp tuple_res = xla::Tuple(input.builder(), {result_token, input});
  xla::XlaOp input_as_result = xla::GetTupleElement(tuple_res, 1);
  return {input_as_result, result_token};
}

RecvResult BuildRecvWithToken(xla::XlaOp token, const xla::Shape& recv_shape,
                              int64_t channel_id) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(channel_id);
  channel_handle.set_type(xla::ChannelHandle::DEVICE_TO_DEVICE);
  xla::XlaOp recv = xla::RecvWithToken(token, recv_shape, channel_handle);
  xla::XlaOp result = xla::GetTupleElement(recv, 0);
  xla::XlaOp new_token = xla::GetTupleElement(recv, 1);
  return {result, new_token};
}

ReduceScatterResult BuildReduceScatter(
    AllReduceType reduce_type, xla::XlaOp input, xla::XlaOp token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout) {
  std::vector<xla::ReplicaGroup> reduce_groups = CreateReduceGroups(groups);
  TokenHandler token_handler(token);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp reduce_result;
  if (pin_layout) {
    torch::lazy::BackendDevice xla_device = GetCurrentDevice();
    xla::Shape reduce_shape = MakeArrayShapeFromDimensions(
        input_shape.dimensions(), input_shape.dynamic_dimensions(),
        input_shape.element_type(),
        static_cast<XlaDeviceType>(xla_device.type()));
    reduce_result = xla::ReduceScatter(
        token_handler.GetInput(input, &input_shape),
        GetReduceComutation(reduce_type, input_shape.element_type()),
        scatter_dim, shard_count, reduce_groups, /*channel_id=*/absl::nullopt,
        /*layout=*/reduce_shape.layout());
  } else {
    reduce_result = xla::ReduceScatter(
        token_handler.GetInput(input, &input_shape),
        GetReduceComutation(reduce_type, input_shape.element_type()),
        scatter_dim, shard_count, reduce_groups);
  }

  if (scale != 1.0) {
    xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
        scale, input_shape.element_type(), input.builder());
    reduce_result = reduce_result * scaling_value;
  }

  return {reduce_result, token_handler.GetNewToken(reduce_result)};
}

}  // namespace torch_xla
