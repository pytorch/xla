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
  using Handle = int64_t;

  // The ShardingContextArena holds per tensor sharding information within
  // a process. This is used to avoid redundant sharding and data transfers.
  class TensorShardingStore {
   public:
    struct ShardingInfo {
      XLATensor::ShardingSpecPtr sharding;
      torch::lazy::Value tensor_node;
    };

    static void RegisterShardingInfo(Handle handle,
                                     ShardingInfo sharding_info) {
      TORCH_LAZY_COUNTER("RegisterShardingInfo", 1);
      std::lock_guard<std::mutex> lock(mutex_);
      tensor_sharding_map_[handle] = sharding_info;
    }
    static void UnregisterShardingInfo(Handle handle) {
      TORCH_LAZY_COUNTER("UnregisterShardingInfo", 1);
      std::lock_guard<std::mutex> lock(mutex_);
      tensor_sharding_map_.erase(handle);
    }

    static const ShardingInfo* GetShardingInfo(Handle handle) {
      // Use find instead of access operator[] for thread-safety.
      auto it = tensor_sharding_map_.find(handle);
      if (it != tensor_sharding_map_.end()) {
        TORCH_LAZY_COUNTER("GetShardingInfoSuccess", 1);
        return &it->second;
      }
      return nullptr;
    }

    static void Reset() {
      tensor_sharding_map_.clear();
    }

   private:
    // A crude locking mechanism for accessing & modifying the static map. We
    // do not distinguish read/write, and there will be only one thread
    // accessing the lock for the most cases.
    // TODO(yeounoh) consider creating a store instance and manage its local
    // sharding info separately, like DeviceContextArena.
    static std::mutex mutex_;
    static std::unordered_map<Handle, ShardingInfo> tensor_sharding_map_;
  };

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

  // Input data nodes require re-sharding if the device data is replicated by
  // the compiler after an execution. This function is intended to be used when
  // there is a sharding annotation attached to the tensor holding the
  // `ir_value`. The returns the updated data node.
  static torch::lazy::Value ShardInputDataNodes(
      torch::lazy::Value ir_value, XLATensor::ShardingSpecPtr sharding_spec);

  // Apply cached tensor data node shardings to the set of nodes and their
  // operands. The shardings are cached during hardInputDataNodes calls. Note
  // that this also resets the TensorShardingStore cache.
  static void ApplyTensorShardingStore(
      std::vector<torch::lazy::Value> ir_values);

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
