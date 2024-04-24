#ifndef XLA_TORCH_XLA_CSRC_XLA_SHARDING_UTIL_H_
#define XLA_TORCH_XLA_CSRC_XLA_SHARDING_UTIL_H_

#include <torch/csrc/jit/python/pybind.h>

#include <tuple>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/tensor.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/service/hlo.pb.h"

namespace torch_xla {

class ShardingUtil {
 public:
  // This maps to `torch_xla.distributed.spmd.ShardingType` enum type.
  enum ShardingType {
    REPLICATED = 0,
    MAXIMAL = 1,
    TUPLE = 2,
    TILED = 3,
    MANUAL = 4,
    PARTIAL = 5,
    UNKNOWN = 6  // implicit replication
  };

  // Determine the ShardingType of the given xla::OpSharding.
  static ShardingType GetShardingType(xla::OpSharding& sharding);

  // Annotates HLO instructions in the lowered computation and returns true if
  // the computation needs to be compiled with SPMD partitioning. For this call
  // to be effective, this needs to be called after the lowering and before
  // building the computation; otherwise, this is a no-op.
  static bool SetHloSharding(LoweringContext* lowering_ctx);

  // Returns true if two sharding specs are the same.
  static bool EqualShardingSpecs(const XLATensor::ShardingSpec& a,
                                 const XLATensor::ShardingSpec& b);

  // Returns true if two OpShardings are the same.
  static bool EqualOpShardings(const xla::OpSharding& a,
                               const xla::OpSharding& b);

  // Creates an xla::OpSharding. `tile_assignmnent` is required for TILED
  // `sharding_type` and `replication_groups` for `PARTIAL`.
  static xla::OpSharding CreateOpSharding(const py::list& tile_assignment,
                                          const py::list& group_assignment,
                                          const py::list& replication_groups,
                                          ShardingType sharding_type);

  // Returns the shape of the resulting shards of `tensor` after applying
  // `sharding`. This assumes the shards will be padded to ensure they all
  // have the same shape.
  static std::vector<int64_t> GetShardShape(
      const XLATensor::ShardingSpecPtr shardings);

  // Uses the provided `sharding` spec and expected shard shape to determine the
  // index slices for the shards which belong on `devices`. Only supports
  // `REPLICATED` and `OTHER` sharding types.
  // For each input device, returns a pair of the shard's replica_id and a
  // vector of TensorIndex denoting the offset of the device's shard into the
  // global tensor.
  static std::vector<std::pair<int, std::vector<at::indexing::TensorIndex>>>
  GetShardReplicaAndIndicesForDevices(const std::vector<int64_t>& shard_shape,
                                      const std::vector<int64_t>& tensor_shape,
                                      const xla::OpSharding sharding,
                                      const std::vector<std::string>& devices);

  // Returns the indices for the shards. Supports `OTHER` sharding types and
  // called when input is sharded along the batch axis.
  static std::vector<std::vector<at::indexing::TensorIndex>>
  GetShardIndicesForMinibatchTensor(const std::vector<int64_t>& shard_shape,
                                    const std::vector<std::string>& devices);

  // Shards a tensor and returns the sharded tensors which belong on `devices`
  // based on the `sharding` spec. REPLICATED sharding should result in shards
  // identical to the input; OTHERS (tiled) sharding result in shards where
  // each data dimension is sharded across devices along the same dimension in
  // the `tile_assignment`; the returned tensor shards vector is
  // indexed by the device IDs. There is no data duplication. Shards are not
  // padded in case the input tensor is not evenly partitionable, unless
  // `padded` is set. The the returned tensors will be in 1:1 correspondence
  // with the `devices` vector, so the `i`th result will belong on the `i`th
  // device.
  static std::vector<at::Tensor> ShardTensor(
      const at::Tensor& tensor, const XLATensor::ShardingSpecPtr shardings,
      const std::vector<std::string>& devices, bool padded = true);

  // Retrieve output sharding of a given XLA computation. ShardingSpec::shape
  // is always on virtual SPMD device.
  static std::vector<XLATensor::ShardingSpecPtr> GetOutputSharding(
      const std::vector<xla::Shape>& output_shapes,
      runtime::ComputationClient::ComputationPtr computation);

  // Create sharded data placeholders, each corresponding to the individual
  // sharding spec from the input list
  static std::vector<torch::lazy::BackendDataPtr> CreateShardedPlaceholder(
      const std::vector<XLATensor::ShardingSpecPtr>& sharding_specs);

  // Prepares output sharding propagation by extracting output parameter
  // ShardingSpec into `sharding_specs` from the SPMD compiled `computation` and
  // placing PjRtShardedData into `data_placeholders`. `data_placeholders`
  // should already contain data placeholders to be used for unsharded output
  // parameters. `tensors` and its `indices` define sync tensors for the
  // outputs.
  static void PrepareOutputShardingPropagation(
      std::vector<XLATensorPtr>* tensors, absl::Span<const size_t> indices,
      runtime::ComputationClient::ComputationPtr computation,
      std::vector<torch::lazy::BackendDataPtr>* data_placeholders,
      std::vector<XLATensor::ShardingSpecPtr>* sharding_specs);

  // Transfers the individual shards to the devices and returns a DataPtr for
  // the PjRtShardedData wrapping the shards.
  static runtime::ComputationClient::DataPtr CreateShardedData(
      const std::vector<at::Tensor>& shards,
      const std::vector<std::string>& devices,
      const XLATensor::ShardingSpecPtr& sharding_spec);

  static void XlaMarkSharding(const at::Tensor& input,
                              xla::OpSharding sharding);

  //////////////////////////// Auto-Sharding ////////////////////////////

  // Construct a device mesh for auto-sharding pass. Returns a tuple of mesh
  // shape and device ids vectors.
  static std::vector<int64_t> GetAutoShardingMesh();
  static std::vector<int64_t> GetAutoShardingMeshIds(
      const xla::HloModuleProto& module);

  // Reshard the parameters if the expected shardings mismatch. Resharding is
  // expensive especially for those already sharded. The cost can easily be
  // armotized over multiple steps, though, since the input sharding is
  // propagated to the output for the subsequent runs. Sharded data transfer
  // during resharding should be asynchronous. It is recommended to keep the
  // input sharding on the input data as-is.
  static void ReshardParameters(
      const xla::HloModuleProto& module, std::vector<XLATensorPtr>* tensors,
      std::vector<torch::lazy::BackendDataPtr>* parameters,
      std::vector<const torch::lazy::Node*>* nodes);

  static void SetAutoSharding();
  static bool GetAutoSharding();

  //////////////////////////// Dynamo Integration ////////////////////////////

  static void XlaMarkShardingDynamoCustomOp(
      const at::Tensor& input, c10::List<at::IntArrayRef> tile_assignment,
      c10::List<at::IntArrayRef> group_assignment,
      c10::List<at::IntArrayRef> replication_groups, int64_t sharding_type);
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_SHARDING_UTIL_H_
