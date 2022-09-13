#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

class ShardingUtil {
 public:
  // Annotate HLO instructions in the lowered compuation by the embedded XLA
  // builder. For this call to be effective, this needs to be called after the
  // lowering and before building the computation; otherwise, this is a no-op.
  static bool SetHloSharding(LoweringContext* lowering_ctx);

  // This is called separately before compilation. This is also useful
  // for debugging partitioned HLO computation and sharding propation.
  static xla::HloModuleProto SpmdPartitioningPass(
      const xla::HloModuleProto& hlo_proto,
      bool conv_halo_exchange_always_on_lhs = true,
      bool choose_faster_windowed_einsum_over_mem = false,
      bool unroll_windowed_einsum = false,
      bool bidirectional_windowed_einsum = false);

  // Shard a tensor and returns the sharded tensors based on the `sharding`
  // spec. REPLICATED sharding should result in shards identical to the input;
  // OTHERS (tiled) sharding result in shards where each data dimension is
  // sharded across devices along the same dimension in the `tile_assignment`;
  // the returned tensor shards vector is indexed by the device IDs. There is no
  // data duplication. Shards are not padded in case the input tensor is not
  // evenly partitionable.
  static std::vector<at::Tensor> ShardTensor(
      const at::Tensor& tensor, const xla::OpSharding sharding,
      const std::vector<std::string>& devices);
};

}  // namespace torch_xla
