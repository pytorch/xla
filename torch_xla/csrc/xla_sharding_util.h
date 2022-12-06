#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "torch/csrc/jit/python/pybind.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

class ShardingUtil {
 public:
  // Annotates HLO instructions in the lowered computation and returns true if
  // the computation needs to be compiled with SPMD partitioning. For this call
  // to be effective, this needs to be called after the lowering and before
  // building the computation; otherwise, this is a no-op.
  static bool SetHloSharding(LoweringContext* lowering_ctx);

  // Returns true if two sharding specs are the same.
  static bool EqualShardingSpecs(const XLATensor::ShardingSpec& a,
                                 const XLATensor::ShardingSpec& b);

  // Create an xla::OpSharding from `tile_assignment` (ndarray).
  static xla::OpSharding CreateOpSharding(const py::list& tile_assignment,
                                          bool replicated = false,
                                          bool manual = false);

  // This is a debugging tool for partitioned HLO generation with different
  // options and sharding propagation.
  static xla::HloModuleProto SpmdPartitioningPass(
      const xla::HloModuleProto& hlo_proto, int64_t num_replicas,
      int64_t num_partitions, bool conv_halo_exchange_always_on_lhs = true,
      bool choose_faster_windowed_einsum_over_mem = false,
      bool unroll_windowed_einsum = false,
      bool bidirectional_windowed_einsum = false);

  // This reshuffles arguments (sharded or replicated) on the devices. The
  // size of the arguments vector must match that of the sharding_specs.
  // TODO(yeounoh) avoiding pre-loading of the unpartitioned input arguments
  // might improve the performance and save the bandwidth.
  static std::vector<std::vector<xla::ComputationClient::DataPtr>> InputHandler(
      std::vector<xla::ComputationClient::DataPtr> arguments,
      std::vector<std::string> devices);

  // Shard a tensor and returns the sharded tensors based on the `sharding`
  // spec. REPLICATED sharding should result in shards identical to the input;
  // OTHERS (tiled) sharding result in shards where each data dimension is
  // sharded across devices along the same dimension in the `tile_assignment`;
  // the returned tensor shards vector is indexed by the device IDs. There is no
  // data duplication. Shards are not padded in case the input tensor is not
  // evenly partitionable, unless `padded` is set.
  static std::vector<at::Tensor> ShardTensor(
      const at::Tensor& tensor, const xla::OpSharding sharding,
      const std::vector<std::string>& devices, bool padded = true);
};

}  // namespace torch_xla
