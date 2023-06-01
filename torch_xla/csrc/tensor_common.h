#ifndef XLA_TORCH_XLA_CSRC_TENSOR_COMMON_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_COMMON_H_

#include <memory>

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

// XLA SPMD sharding spec annoation. The XLA tensor uses this to create
// HloSharding for replication, manual and tile shardings.
struct ShardingSpec {
  ShardingSpec(const xla::OpSharding& sharding) : sharding(sharding) {}
  ShardingSpec(const xla::OpSharding& sharding, const xla::Shape& shape)
      : sharding(sharding), shape(shape) {}

  xla::OpSharding sharding;
  // Optional source tensor shape unpartitioned.
  std::optional<xla::Shape> shape;
};

using ShardingSpecPtr = std::shared_ptr<ShardingSpec>;
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_COMMON_H_