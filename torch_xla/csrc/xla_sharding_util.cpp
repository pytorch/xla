#include "torch_xla/csrc/xla_sharding_util.h"

#include <ATen/TensorIndexing.h>

#include <cmath>
#include <unordered_map>

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

using xla::internal::XlaBuilderFriend;

// Extract dimensions of the nested input array/list. For instance, an input 2D
// list, [[1, 2, 3], [4, 5, 6]] has [2, 3] dimensions with 2 rows and 3 columns.
std::vector<int64_t> TileAssignmentDimensions(
    const py::list& tile_assignments) {
  std::vector<int64_t> dims;
  py::list r = tile_assignments;
  while (true) {
    XLA_CHECK(r.size() > 0)
        << "Invalid argument: empty list is not a valid element type.";
    dims.push_back(r.size());
    auto type = r[0].attr("__class__").attr("__name__").cast<std::string>();
    if (type == "list") {
      r = r[0];
    } else if ((type != "int") && (type != "float")) {
      TF_LOG(ERROR) << "Invalid arguments: element type " << type;
    } else {
      break;
    }
  }
  return dims;
}

}  // namespace

bool ShardingUtil::SetHloSharding(LoweringContext* lowering_ctx) {
  bool is_sharded = false;
  for (std::pair<torch::lazy::Output, xla::XlaOp> elem :
       lowering_ctx->GetEmittedOutputs()) {
    const torch::lazy::Node* node = elem.first.node;
    const XlaNode* xla_node = dynamic_cast<const XlaNode*>(node);
    auto instruction = XlaBuilderFriend::GetInstruction(elem.second);
    if (xla_node->GetSharding() != nullptr) {
      *instruction->mutable_sharding() = *xla_node->GetSharding();
      is_sharded = true;
    }
  }
  return is_sharded;
}

bool ShardingUtil::EqualShardingSpecs(const XLATensor::ShardingSpec& a,
                                      const XLATensor::ShardingSpec& b) {
  return xla::protobuf_util::ProtobufEquals(a.sharding, b.sharding);
}

xla::OpSharding ShardingUtil::CreateOpSharding(const py::list& tile_assignment,
                                               bool replicated, bool manual) {
  XLA_CHECK(!(replicated && manual))
      << "Invalid arguments: replicated=" << replicated
      << ", manual=" << manual;

  xla::OpSharding sharding;
  if (replicated) {
    sharding = xla::HloSharding::Replicate().ToProto();
  } else if (manual) {
    sharding = xla::HloSharding::Manual().ToProto();
  } else {
    // Sharding type is tiled
    auto dims = TileAssignmentDimensions(tile_assignment);
    xla::Array<int64_t> tile_array(dims);
    switch (dims.size()) {
      case 1:
        tile_array.Each([&](absl::Span<const int64_t> indices, int64_t* v) {
          *v = tile_assignment[indices[0]].cast<int64_t>();
        });
        break;
      case 2:
        tile_array.Each([&](absl::Span<const int64_t> indices, int64_t* v) {
          auto r = tile_assignment[indices[0]].cast<py::list>();
          *v = r[indices[1]].cast<int64_t>();
        });
        break;
      case 3:
        tile_array.Each([&](absl::Span<const int64_t> indices, int64_t* v) {
          auto r = tile_assignment[indices[0]].cast<py::list>();
          r = r[indices[1]].cast<py::list>();
          *v = r[indices[2]].cast<int64_t>();
        });
        break;
      case 4:
        tile_array.Each([&](absl::Span<const int64_t> indices, int64_t* v) {
          auto r = tile_assignment[indices[0]].cast<py::list>();
          r = r[indices[1]].cast<py::list>();
          r = r[indices[2]].cast<py::list>();
          *v = r[indices[3]].cast<int64_t>();
        });
        break;
      case 5:
        tile_array.Each([&](absl::Span<const int64_t> indices, int64_t* v) {
          auto r = tile_assignment[indices[0]].cast<py::list>();
          r = r[indices[1]].cast<py::list>();
          r = r[indices[2]].cast<py::list>();
          r = r[indices[3]].cast<py::list>();
          *v = r[indices[4]].cast<int64_t>();
        });
        break;
      default:
        TF_LOG(ERROR) << "Invalid arguments: tile_assignment ranks > 5";
    }
    xla::HloSharding hlo_sharding = xla::HloSharding::Tile(tile_array);
    sharding = hlo_sharding.ToProto();
  }
  return sharding;
}

xla::HloModuleProto ShardingUtil::SpmdPartitioningPass(
    const xla::HloModuleProto& hlo_proto, int64_t num_replicas,
    int64_t num_partitions, bool conv_halo_exchange_always_on_lhs,
    bool choose_faster_windowed_einsum_over_mem, bool unroll_windowed_einsum,
    bool bidirectional_windowed_einsum) {
  // TODO(yeounoh) propagate this down to the PJRT client
  auto execution_options = xla::CreateDefaultExecutionOptions();
  execution_options.set_use_spmd_partitioning(true);
  execution_options.set_num_replicas(num_replicas);
  execution_options.set_num_partitions(num_partitions);
  auto module_config = xla::HloModule::CreateModuleConfigFromProto(
                           hlo_proto, xla::DebugOptions(), &execution_options)
                           .value();
  auto module = xla::HloModule::CreateFromProto(hlo_proto, module_config,
                                                /*prohibit_empty_literal=*/true)
                    .value();

  xla::spmd::SpmdPartitionerOptions options;
  options.conv_halo_exchange_always_on_lhs = conv_halo_exchange_always_on_lhs;
  options.allow_module_signature_change = true;
  options.choose_faster_windowed_einsum_over_mem =
      choose_faster_windowed_einsum_over_mem;
  options.unroll_windowed_einsum = unroll_windowed_einsum;
  options.bidirectional_windowed_einsum = bidirectional_windowed_einsum;

  xla::HloPassPipeline pass("spmd-partitioning");
  pass.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                 /*allow_mixed_precision=*/false);
  // TODO(yeounoh) side-effecting ops gets assigned replicated sharding.
  pass.AddPass<xla::ShardingPropagation>(
      /*is_spmd=*/true, /*propagate_metadata=*/false,
      /*allow_spmd_sharding_propagation_to_output=*/
      absl::MakeConstSpan({true}));
  pass.AddPass<xla::spmd::SpmdPartitioner>(
      /*num_partitions=*/num_partitions,
      /*num_replicas=*/num_replicas, options,
      xla::spmd::GetDefaultCollectiveOpsCreator(
          /*num_partitions=*/num_partitions,
          /*num_replicas=*/num_replicas));
  pass.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                 /*allow_mixed_precision=*/false);
  const auto& pass_status = pass.Run(module.get());
  if (!pass_status.ok()) {
    XLA_ERROR() << "spmd-partitioning pass failed";
  }

  return module.get()->ToProto();
}

// Builds a map from the device's global ordinal to its index in the `devices`
// array. This is used by `ShardTensor` and `InputHandler` to ensure the
// order of the output corresponds to the order of the `devices`, which can be
// arbitrarily set by the caller.
static std::unordered_map<int, int> build_index_map(
    const std::vector<std::string>& devices) {
  std::unordered_map<int, int> device_index;
  for (int i = 0; i < devices.size(); ++i) {
    int global_ordinal = ParseDeviceString(devices[i]).ordinal();
    device_index[global_ordinal] = i;
  }
  return device_index;
}

std::vector<std::vector<xla::ComputationClient::DataPtr>>
ShardingUtil::InputHandler(
    std::vector<xla::ComputationClient::DataPtr> arguments,
    std::vector<std::string> devices) {
  std::vector<std::vector<xla::ComputationClient::DataPtr>> arguments_by_device(
      devices.size(),
      std::vector<xla::ComputationClient::DataPtr>(arguments.size()));
  auto device_index = build_index_map(devices);

  for (int64_t argument_i = 0; argument_i < arguments.size(); ++argument_i) {
    auto shards =
        xla::ComputationClient::Get()->GetDataShards(arguments[argument_i]);
    if (shards.size() > 1) {
      // Input is sharded across addressable devices
      for (auto shard : shards) {
        int global_ordinal = ParseDeviceString(shard->device()).ordinal();
        int device_i = device_index[global_ordinal];
        arguments_by_device[device_i][argument_i] = shard;
      }
    } else {
      // Input is replicated across addressable devices
      int global_ordinal = ParseDeviceString(shards[0]->device()).ordinal();
      int source_device_i = device_index[global_ordinal];
      arguments_by_device[source_device_i][argument_i] = shards[0];
      for (int64_t device_i = 0; device_i < devices.size(); ++device_i) {
        if (device_i != source_device_i) {
          arguments_by_device[device_i][argument_i] =
              xla::ComputationClient::Get()->CopyToDevice(shards[0],
                                                          devices[device_i]);
        }
      }
    }
  }

  return arguments_by_device;
}

std::vector<at::Tensor> ShardingUtil::ShardTensor(
    const at::Tensor& tensor, const xla::OpSharding sharding,
    const std::vector<std::string>& devices, bool padded) {
  TF_LOG(INFO) << "ShardTensor with sharding type(" << sharding.type() << ")..."
               << std::endl;
  auto device_index = build_index_map(devices);
  std::vector<at::Tensor> shards(devices.size());
  if (sharding.type() == xla::OpSharding::REPLICATED) {
    std::fill_n(shards.begin(), shards.size(), tensor);
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    XLA_CHECK(sharding.tile_shape().dimensions_size() <= 2);
    XLA_CHECK(tensor.sizes().size() >= sharding.tile_shape().dimensions_size());

    auto tile_shape = sharding.tile_assignment_dimensions();

    // `partition_len[j]` is the size of dimension `j` in the resulting shard.
    std::vector<int64_t> partition_len;
    for (int j = 0; j < tile_shape.size(); j++) {
      partition_len.push_back(tensor.sizes()[j] / tile_shape[j] +
                              (tensor.sizes()[j] % tile_shape[j] != 0));
    }

    for (size_t i = 0; i < sharding.tile_assignment_devices().size(); i++) {
      int64_t core = sharding.tile_assignment_devices()[i];
      if (device_index.find(core) == device_index.end()) {
        // Skip any shards whose device is not part of the `devices` list.
        continue;
      }

      // Given the shard's row-major index `i`, we need to calculate shard's
      // coordinates (n_0, ..., n_d) in the tiling to generate the index slices.
      // Using `N_j = tile_shape[j]` and `0 <= n_j < N_j`, the following
      // equation needs to be solved for all n_j:
      //            `i = n_d + N_d * (n_{d-1} + N_{d-1} * (... + (N_1 * n_0)))`
      // Let `offset_j = n_j + N_j * (n_{j-1} + N_{j-1} * (... + (N_1 * n_0)))`.
      // Then `offset_d = i`, `n_j = offset_j % N_j`, and `offset_{j-1} =
      // offset_j / N_j`.
      int offset = i;
      std::vector<at::indexing::TensorIndex> indices;
      for (int j = tile_shape.size() - 1; j >= 0; j--) {
        int64_t n_j = offset % tile_shape[j];
        auto slice = at::indexing::Slice(n_j * partition_len[j],
                                         (n_j + 1) * partition_len[j]);
        indices.push_back(at::indexing::TensorIndex(slice));
        offset /= tile_shape[j];
      }
      std::reverse(indices.begin(), indices.end());
      at::Tensor shard =
          tensor.index(c10::ArrayRef<at::indexing::TensorIndex>(indices));

      shards[device_index[core]] =
          shard.contiguous(at::MemoryFormat::Contiguous);
    }

    // Zero-pad to the right to ensure the sizes are even
    if (shards.size() > 0 && padded) {
      for (size_t i = 0; i < shards.size(); ++i) {
        std::vector<long> pads;
        for (size_t j = 0; j < partition_len.size(); ++j) {
          XLA_CHECK_GE(partition_len[j], shards[i].sizes().at(j));
          pads.push_back(partition_len[j] - shards[i].sizes().at(j));
          pads.push_back(0);  // no padding on lhs
        }
        // Padding starts from the last dimension
        std::reverse(pads.begin(), pads.end());
        shards[i] = at::constant_pad_nd(
            shards[i], c10::IntArrayRef(pads.data(), pads.size()), 0);
      }
    }
  } else if ((sharding.type() == xla::OpSharding::MANUAL) ||
             (sharding.type() == xla::OpSharding::TUPLE)) {
    TF_LOG(ERROR) << "Unsupported OpSharding type " << sharding.type();
  }
  return shards;
}

}  // namespace torch_xla
