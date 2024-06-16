#ifndef XLA_TORCH_XLA_CSRC_CROSS_REPLICA_REDUCES_H_
#define XLA_TORCH_XLA_CSRC_CROSS_REPLICA_REDUCES_H_

#include <vector>

#include "absl/types/span.h"
#include "torch/csrc/lazy/core/ir.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "xla/client/xla_builder.h"

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

struct AllGatherResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

struct AllGatherResultCoalesced {
  std::vector<xla::XlaOp> result;
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

struct ReduceScatterResult {
  xla::XlaOp result;
  xla::XlaOp token;
};

struct ReduceScatterResultCoalesced {
  std::vector<xla::XlaOp> result;
  xla::XlaOp token;
};

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> operands,
    xla::XlaOp token, double scale,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout);

xla::XlaOp BuildAllReduce(AllReduceType reduce_type, xla::XlaOp operand,
                          double scale,
                          const std::vector<std::vector<int64_t>>& groups);

AllToAllResult BuildAllToAll(xla::XlaOp input, xla::XlaOp token,
                             int64_t split_dimension, int64_t concat_dimension,
                             int64_t split_count,
                             const std::vector<std::vector<int64_t>>& groups,
                             bool pin_layout);

AllGatherResult BuildAllGather(xla::XlaOp input, xla::XlaOp token, int64_t dim,
                               int64_t shard_count,
                               const std::vector<std::vector<int64_t>>& groups,
                               bool pin_layout);

AllGatherResultCoalesced BuildAllGatherCoalesced(
    absl::Span<const xla::XlaOp> inputs, xla::XlaOp token, int64_t dim,
    int64_t shard_count, const std::vector<std::vector<int64_t>>& groups,
    bool pin_layout);

CollectivePermuteResult BuildCollectivePermute(
    xla::XlaOp input, xla::XlaOp token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs);

SendResult BuildSendWithToken(xla::XlaOp input, xla::XlaOp token,
                              int64_t channel_id);

RecvResult BuildRecvWithToken(xla::XlaOp token, const xla::Shape& recv_shape,
                              int64_t channel_id);

ReduceScatterResult BuildReduceScatter(
    AllReduceType reduce_type, xla::XlaOp input, xla::XlaOp token, double scale,
    int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout);

xla::XlaOp BuildReduceScatter(AllReduceType reduce_type, xla::XlaOp input,
                              double scale, int64_t scatter_dim,
                              int64_t shard_count,
                              const std::vector<std::vector<int64_t>>& groups);

ReduceScatterResultCoalesced BuildReduceScatterCoalesced(
    AllReduceType reduce_type, absl::Span<const xla::XlaOp> inputs,
    xla::XlaOp token, double scale, int64_t scatter_dim, int64_t shard_count,
    const std::vector<std::vector<int64_t>>& groups, bool pin_layout);

std::vector<torch::lazy::Value> GetOperandListWithToken(
    c10::ArrayRef<torch::lazy::Value> operands,
    const torch::lazy::Value& token);

const torch::lazy::Value& GetAllReduceToken(
    const torch::lazy::BackendDevice& device);
void SetAllReduceToken(const torch::lazy::BackendDevice& device,
                       const std::shared_ptr<torch::lazy::Value>& token);

AllReduceType GetReduceType(std::string_view reduce_type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_CROSS_REPLICA_REDUCES_H_
