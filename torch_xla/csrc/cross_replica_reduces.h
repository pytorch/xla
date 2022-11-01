#pragma once

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
// #include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

enum class AllReduceType {
  kSum,
  kMin,
  kMax,
  kMul,
  kOr,
  kAnd,
};

struct AllToAllResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

struct CollectivePermuteResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

struct SendResult {
  xla::XlaOp input_as_result;
  xla::XlaOp token;
};

struct RecvResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> operands,
    xla::XlaOp token, double scale,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout);

AllToAllResult BuildAllToAll(xla::XlaOp input, xla::XlaOp token,
                             int64_t split_dimension, int64_t concat_dimension,
                             int64_t split_count,
                             const std::vector<std::vector<int64_t>>& groups,
                             bool pin_layout);

std::vector<xla::XlaOp> BuildAllGather(
    absl::Span<const xla::XlaOp>, xla::XlaOp token, int64_t dim,
    int64_t shard_count, const std::vector<std::vector<int64_t>>& groups,
    bool pin_layout);

CollectivePermuteResult BuildCollectivePermute(
    xla::XlaOp input, xla::XlaOp token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs);

SendResult BuildSendWithToken(xla::XlaOp input, xla::XlaOp token,
                              int64_t channel_id);

RecvResult BuildRecvWithToken(xla::XlaOp token, const xla::Shape& recv_shape,
                              int64_t channel_id);

std::vector<xla::XlaOp> BuildReduceScatter(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> inputs,
    xla::XlaOp token, double scale, int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout);

std::vector<torch::lazy::Value> GetOperandList(
    c10::ArrayRef<torch::lazy::Value> operands,
    const torch::lazy::Value& token);

}  // namespace torch_xla
