#pragma once

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

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

struct ReduceScatterResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> operands,
    xla::XlaOp token, double scale,
    const std::vector<std::vector<xla::int64_t>>& groups);

AllToAllResult BuildAllToAll(
    xla::XlaOp input, xla::XlaOp token, xla::int64_t split_dimension,
    xla::int64_t concat_dimension, xla::int64_t split_count,
    const std::vector<std::vector<xla::int64_t>>& groups);

CollectivePermuteResult BuildCollectivePermute(
    xla::XlaOp input, xla::XlaOp token,
    const std::vector<std::pair<xla::int64_t, xla::int64_t>>& source_target_pairs);

ReduceScatterResult BuildReduceScatter(
    AllReduceType reduce_type, xla::XlaOp input, xla::XlaOp token, double scale,
    xla::int64 scatter_dim, xla::int64 shard_count,
    const std::vector<std::vector<xla::int64>>& groups);

}  // namespace torch_xla
