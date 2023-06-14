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
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

using tsl::ERROR;
using tsl::INFO;
using xla::internal::XlaBuilderFriend;

// Return py::obj type as string.
std::string GetPyType(const py::object& elem) {
  return elem.attr("__class__").attr("__name__").cast<std::string>();
}

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
    std::string type = GetPyType(r[0]);
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

// Builds a map from the device's global ordinal to its index in the `devices`
// array. This is used by `ShardTensor` and `InputHandler` to ensure the
// order of the output corresponds to the order of the `devices`, which can be
// arbitrarily set by the caller.
std::unordered_map<int, int> build_index_map(
    const std::vector<std::string>& devices) {
  std::unordered_map<int, int> device_index;
  for (int i = 0; i < devices.size(); ++i) {
    int global_ordinal = ParseDeviceString(devices[i]).ordinal();
    device_index[global_ordinal] = i;
  }
  return device_index;
}

xla::Array<int64_t> TileListToArray(const py::list& tile_assignment) {
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
  return tile_array;
}

// Extract a list view of device IDs as group members per replication group.
std::vector<std::vector<int64_t>> ExtractGroupMembers(
    const py::list& replication_groups) {
  std::vector<std::vector<int64_t>> groups;
  groups.reserve(replication_groups.size());
  for (int i = 0; i < replication_groups.size(); ++i) {
    std::string type = GetPyType(replication_groups[i]);
    XLA_CHECK(type == "list")
        << "Invalid replication group type: list is expected, got " << type;
    const py::list& group = replication_groups[i];
    std::vector<int64_t> group_members;
    group_members.reserve(group.size());
    for (int j = 0; j < group.size(); ++j) {
      try {
        group_members.push_back(group[j].cast<int64_t>());
      } catch (py::error_already_set& e) {
        TF_LOG(ERROR) << "Invalid arguments: element type "
                      << GetPyType(group[j]);
      }
    }
    groups.push_back(group_members);
  }
  return groups;
}

}  // namespace

bool ShouldUseVirtualDevice() {
  bool use_virtual_device =
      runtime::sys_util::GetEnvBool("XLA_USE_SPMD", false);
  if (use_virtual_device) {
    TF_LOG(INFO) << "Using SPMD virtual device optimization";
  }
  return use_virtual_device;
}

bool ShardingUtil::UseVirtualDevice() {
  static bool use_virtual_device = ShouldUseVirtualDevice();
  return use_virtual_device;
}

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

ShardingUtil::ShardingType ShardingUtil::GetShardingType(
    xla::OpSharding& sharding) {
  switch (sharding.type()) {
    case xla::OpSharding::REPLICATED:
      return ShardingType::REPLICATED;
    case xla::OpSharding::MAXIMAL:
      return ShardingType::MAXIMAL;
    case xla::OpSharding::TUPLE:
      return ShardingType::TUPLE;
    case xla::OpSharding::OTHER:
      // OTHER sharding can indicate either PARTIAL or TILED sharding.
      return sharding.replicate_on_last_tile_dim() ? ShardingType::PARTIAL
                                                   : ShardingType::TILED;
    case xla::OpSharding::MANUAL:
      return ShardingType::MANUAL;
    default:
      TF_LOG(ERROR) << "Unsupported sharding type: " << sharding.type();
  }
}

bool ShardingUtil::EqualShardingSpecs(const XLATensor::ShardingSpec& a,
                                      const XLATensor::ShardingSpec& b) {
  return xla::protobuf_util::ProtobufEquals(a.sharding, b.sharding);
}

xla::OpSharding ShardingUtil::CreateOpSharding(
    const py::list& tile_assignment, const py::list& group_assignment,
    const py::list& replication_groups, ShardingType sharding_type) {
  xla::OpSharding sharding;
  switch (sharding_type) {
    case ShardingType::MANUAL: {
      TF_LOG(ERROR) << "Invalid arguments: sharding_type (MANUAL) is "
                    << "currently not supported";
      break;
    }
    case ShardingType::TUPLE: {
      TF_LOG(ERROR) << "Invalid arguments: sharding_type (TUPLE) is "
                    << "currently not supported";
      break;
    }
    // REPLICATED reduces to MAXIMAL in case of a single device.
    case ShardingType::MAXIMAL:
    case ShardingType::REPLICATED: {
      sharding = xla::HloSharding::Replicate().ToProto();
      break;
    }
    case ShardingType::TILED: {
      xla::Array<int64_t> tile_array = TileListToArray(tile_assignment);
      xla::HloSharding hlo_sharding = xla::HloSharding::Tile(tile_array);
      sharding = hlo_sharding.ToProto();
      break;
    }
    case ShardingType::PARTIAL: {
      XLA_CHECK(replication_groups.size() > 0)
          << "ShardingType.PARTIAL requires non-empty replication groups.";
      xla::Array<int64_t> group_tiling = TileListToArray(group_assignment);
      auto group_members = ExtractGroupMembers(replication_groups);
      std::vector<absl::Span<const int64_t>> group_members_view;
      group_members_view.reserve(group_members.size());
      for (auto& group : group_members) {
        auto group_view = absl::MakeConstSpan(group);
        group_members_view.push_back(group_view);
      }
      XLA_CHECK(group_tiling.num_elements() == group_members_view.size());

      sharding = xla::HloSharding::PartialTile(group_tiling, group_members_view)
                     .ToProto();
      break;
    }
    default: {
      TF_LOG(ERROR) << "Invalid arguments: sharding_type " << sharding_type;
    }
  }
  TF_VLOG(INFO) << "OpSharding (ShardingType: " << sharding_type << "):\n"
                << sharding.DebugString();
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

std::vector<std::vector<runtime::ComputationClient::DataPtr>>
ShardingUtil::InputHandler(
    std::vector<runtime::ComputationClient::DataPtr> arguments,
    std::vector<std::string> devices) {
  std::vector<std::vector<runtime::ComputationClient::DataPtr>>
      arguments_by_device(
          devices.size(),
          std::vector<runtime::ComputationClient::DataPtr>(arguments.size()));
  // This assumes that the (local) devices are sorted, in order to associate
  // the first local index with the first global device ordinal.
  auto device_index = build_index_map(devices);

  for (int64_t argument_i = 0; argument_i < arguments.size(); ++argument_i) {
    auto shards =
        runtime::GetComputationClient()->GetDataShards(arguments[argument_i]);
    // With SPMD execution, all input is distributed across addressable devices,
    // either by sharding or replication.
    for (auto shard : shards) {
      int global_ordinal = ParseDeviceString(shard->device()).ordinal();
      int device_i = device_index[global_ordinal];
      arguments_by_device[device_i][argument_i] = shard;
    }
  }

  return arguments_by_device;
}

std::vector<runtime::ComputationClient::DataPtr> ShardingUtil::OutputHandler(
    std::vector<std::vector<runtime::ComputationClient::DataPtr>>
        sharded_results,
    std::vector<XLATensor::ShardingSpecPtr> sharding_specs,
    bool replicated_output) {
  std::vector<runtime::ComputationClient::DataPtr> outputs;
  outputs.reserve(sharding_specs.size());
  for (int i = 0; i < sharding_specs.size(); ++i) {
    XLATensor::ShardingSpecPtr sharding = sharding_specs[i];
    if (replicated_output && sharding &&
        (sharding->sharding.type() != xla::OpSharding::REPLICATED)) {
      XLA_CHECK(sharding->shape.has_value())
          << "Sharding or Wrapping data shards in OutputHandler requires "
             "unpartitioned tensor shape.";
      // Reshards replicated output if `sharding` is present.
      std::vector<at::Tensor> tensors = XlaDataToTensors(
          {WrapXlaData(sharded_results[0][i])},
          TensorTypeFromXlaType(sharding->shape.value().element_type()));
      outputs.push_back(UnwrapXlaData(CreateTensorsData(
          tensors, {sharding},
          std::vector<std::string>{GetVirtualDevice().toString()})[0]));
    } else {
      // The output is sharded or replicated.
      std::vector<runtime::ComputationClient::DataPtr> shards;
      shards.reserve(sharded_results.size());
      for (int j = 0; j < sharded_results.size(); ++j) {
        XLA_CHECK(sharded_results[j][i]->HasValue());
        shards.push_back(sharded_results[j][i]);
      }
      if (!sharding) {
        // Without an explicit sharding annotation, the output is implicitly
        // replicated
        sharding = std::make_shared<XLATensor::ShardingSpec>(
            xla::HloSharding::Replicate().ToProto(),
            sharded_results[0][i]->shape());
      }
      outputs.push_back(runtime::GetComputationClient()->WrapDataShards(
          shards, GetVirtualDevice().toString(), sharding->shape.value(),
          sharding->sharding));
    }
  }
  return outputs;
}

std::vector<int64_t> ShardingUtil::GetShardShape(
    const at::Tensor& tensor, const xla::OpSharding sharding) {
  if (sharding.type() == xla::OpSharding::REPLICATED) {
    return tensor.sizes().vec();
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    auto tile_shape = sharding.tile_assignment_dimensions();

    // `shard_shape[j]` is the size of dimension `j` in the resulting shard.
    std::vector<int64_t> shard_shape;
    for (int j = 0; j < tile_shape.size(); j++) {
      if (sharding.replicate_on_last_tile_dim() && j == tile_shape.size() - 1) {
        continue;
      }
      shard_shape.push_back(tensor.sizes()[j] / tile_shape[j] +
                            (tensor.sizes()[j] % tile_shape[j] != 0));
    }
    return shard_shape;
  } else {
    TF_LOG(ERROR) << "Unsupported OpSharding type " << sharding.type();
  }
}

std::vector<std::vector<at::indexing::TensorIndex>>
ShardingUtil::GetShardIndicesForDevices(
    const std::vector<int64_t>& shard_shape,
    const std::vector<int64_t>& tensor_shape, const xla::OpSharding sharding,
    const std::vector<std::string>& devices) {
  // `shard_indices[dev][dim]` represents the index slice for dimension `dim`
  // that belongs on device `devices[dev]` if the tensor is sharded. If
  // `sharding` is REPLICATED, `shard_indices[dev]` will only have a single
  // Ellipsis element to indicate that the tensor is replicated across all
  // dimensions.
  std::vector<std::vector<at::indexing::TensorIndex>> shard_indices(
      devices.size());
  auto tile_shape = sharding.tile_assignment_dimensions();
  if (sharding.type() == xla::OpSharding::REPLICATED) {
    // Use Ellipsis to indicate all dimensions are replicated
    auto ellipsis = at::indexing::TensorIndex(at::indexing::Ellipsis);
    auto indices = std::vector<at::indexing::TensorIndex>({ellipsis});
    std::fill_n(shard_indices.begin(), shard_indices.size(), indices);
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    auto device_index = build_index_map(devices);
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
        if (sharding.replicate_on_last_tile_dim() &&
            j == tile_shape.size() - 1) {
          // the last tile assignment dimension is replicated, which implies
          // that the consecutive `tile_shape[j]` devices hold the replicated.
          offset /= tile_shape[j];
          continue;
        }
        int64_t n_j = offset % tile_shape[j];
        // Clamp the slice bounds to the tensor shape to accurately reflect
        // the shard size without padding.
        int start = std::min(n_j * shard_shape[j], tensor_shape[j]);
        int end = std::min((n_j + 1) * shard_shape[j], tensor_shape[j]);
        auto slice = at::indexing::Slice(start, end);
        indices.push_back(at::indexing::TensorIndex(slice));
        offset /= tile_shape[j];
      }
      std::reverse(indices.begin(), indices.end());
      shard_indices[device_index[core]] = indices;
    }
  } else {
    TF_LOG(ERROR) << "Unsupported OpSharding type " << sharding.type();
  }
  return shard_indices;
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

    auto shard_shape = GetShardShape(tensor, sharding);
    auto shard_indices = GetShardIndicesForDevices(
        shard_shape, tensor.sizes().vec(), sharding, devices);

    for (size_t i = 0; i < shard_indices.size(); i++) {
      at::Tensor shard = tensor.index(
          c10::ArrayRef<at::indexing::TensorIndex>(shard_indices[i]));
      shards[i] = shard.contiguous(at::MemoryFormat::Contiguous);
    }

    // Zero-pad to the right to ensure the sizes are even
    if (shards.size() > 0 && padded) {
      for (size_t i = 0; i < shards.size(); ++i) {
        std::vector<long> pads;
        for (size_t j = 0; j < shard_shape.size(); ++j) {
          XLA_CHECK_GE(shard_shape[j], shards[i].sizes().at(j));
          pads.push_back(shard_shape[j] - shards[i].sizes().at(j));
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

void ShardingUtil::PrepareOutputShardingPropagation(
    std::vector<XLATensorPtr>* tensors, absl::Span<const size_t> indices,
    ComputationPtr computation,
    std::vector<torch::lazy::BackendDataPtr>* data_placeholders,
    std::vector<XLATensor::ShardingSpecPtr>* sharding_specs) {
  // Resizes the containers to `indices.size()`.
  data_placeholders->resize(indices.size());
  sharding_specs->resize(indices.size());

  auto computation_proto = computation->computation().proto();

  std::vector<xla::OpSharding> output_shardings;
  if (computation_proto.has_spmd_output_sharding()) {
    if (computation_proto.spmd_output_sharding().tuple_shardings().size() > 0) {
      auto tuple_shardings =
          computation_proto.spmd_output_sharding().tuple_shardings();
      output_shardings = std::vector<xla::OpSharding>(tuple_shardings.begin(),
                                                      tuple_shardings.end());
    } else {
      output_shardings = std::vector<xla::OpSharding>{
          computation_proto.spmd_output_sharding()};
    }
  }

  // Output parameter sharding annotations, defaults to REPLICATED(0) if unset.
  if (output_shardings.empty()) {
    // Initializes with default sharding type, REPLCIATED.
    output_shardings.resize(indices.size());
  }
  XLA_CHECK(indices.size() == output_shardings.size())
      << "Expected size: " << indices.size()
      << ", actual size: " << output_shardings.size();

  for (int i = 0; i < indices.size(); ++i) {
    auto xtensor = (*tensors)[indices[i]];
    if (output_shardings[i].type()) {
      // Tensor sharding annotation type is non-zero (sharded).
      (*sharding_specs)[i] = std::make_shared<XLATensor::ShardingSpec>(
          output_shardings[i],
          MakeShapeWithDeviceLayout(
              xtensor->shape().get(),
              static_cast<XlaDeviceType>(xtensor->GetDevice().type())));
      xtensor->SetShardingSpec(*(*sharding_specs)[i]);
    } else {
      // Clear sharding if the output parameter is no longer sharded, this
      // assumes that the output is implicitly replicated and wrapped inside
      // PjRtShardedData.
      (*sharding_specs)[i] = std::make_shared<XLATensor::ShardingSpec>(
          xla::HloSharding::Replicate().ToProto(),
          MakeShapeWithDeviceLayout(
              xtensor->shape().get(),
              static_cast<XlaDeviceType>(xtensor->GetDevice().type())));
      xtensor->ClearShardingSpec();
    }

    // Create sharded data placeholder, this will be used to
    // hold the corresponding computation results for both sharding &
    // replication.
    auto sharded_data_placeholder =
        WrapXlaData(runtime::GetComputationClient()->WrapDataShards(
            {}, GetVirtualDevice().toString(),
            (*sharding_specs)[i]->shape.value(),
            (*sharding_specs)[i]->sharding));

    // Register the sharded data placeholder to the tensor and its node.
    (*data_placeholders)[i] = sharded_data_placeholder;
    xtensor->data()->handle = (*data_placeholders)[i];
    if (auto ir_value = xtensor->CurrentIrValue()) {
      if (DeviceData* device_data_node =
              DeviceData::Cast(ir_value.node.get())) {
        device_data_node->Assign(xtensor->data()->handle);
      }
    }
  }
}

void ShardingUtil::PrepareOutputShardingPropagation(
    std::vector<torch::lazy::BackendDataPtr>& placeholders,
    std::vector<XLATensor::ShardingSpecPtr>& sharding_specs,
    std::vector<xla::Shape>* output_shapes, ComputationPtr computation,
    const torch::lazy::BackendDevice& device) {
  auto computation_proto = computation->computation().proto();
  std::vector<xla::OpSharding> output_shardings;
  if (computation_proto.has_spmd_output_sharding()) {
    if (computation_proto.spmd_output_sharding().tuple_shardings().size() > 0) {
      auto tuple_shardings =
          computation_proto.spmd_output_sharding().tuple_shardings();
      output_shardings = std::vector<xla::OpSharding>(tuple_shardings.begin(),
                                                      tuple_shardings.end());
    } else {
      output_shardings = std::vector<xla::OpSharding>{
          computation_proto.spmd_output_sharding()};
    }
  }

  // Output parameter sharding annotations, defaults to REPLICATED(0) if
  // unset.
  if (output_shardings.empty()) {
    // Initializes with default sharding type, REPLCIATED.
    output_shardings.resize(placeholders.size());
  }

  for (int i = 0; i < placeholders.size(); ++i) {
    if (output_shardings[i].type()) {
      // Tensor sharding annotation type is non-zero (sharded).
      sharding_specs[i] = std::make_shared<XLATensor::ShardingSpec>(
          output_shardings[i],
          MakeShapeWithDeviceLayout((*output_shapes)[i],
                                    static_cast<XlaDeviceType>(device.type())));
    } else {
      // Clear sharding if the output parameter is no longer sharded, this
      // assumes that the output is implicitly replicated and wrapped inside
      // PjRtShardedData.
      sharding_specs[i] = std::make_shared<XLATensor::ShardingSpec>(
          xla::HloSharding::Replicate().ToProto(),
          MakeShapeWithDeviceLayout((*output_shapes)[i],
                                    static_cast<XlaDeviceType>(device.type())));
    }

    // Create sharded data placeholder, this will be used to
    // hold the corresponding computation results for both sharding &
    // replication.
    auto sharded_data_placeholder =
        WrapXlaData(runtime::GetComputationClient()->WrapDataShards(
            {}, GetVirtualDevice().toString(), sharding_specs[i]->shape.value(),
            sharding_specs[i]->sharding));

    // Register the sharded data placeholder to the tensor and its node.
    placeholders[i] = sharded_data_placeholder;
  }
}

runtime::ComputationClient::DataPtr ShardingUtil::CreateShardedData(
    std::vector<at::Tensor>& local_shards, std::vector<std::string>& devices,
    xla::Shape global_shape, xla::OpSharding sharding) {
  XLA_CHECK(local_shards.size() == devices.size())
      << "A device must be speficied for each shard";
  std::vector<runtime::ComputationClient::TensorSource> source_tensors;
  for (int64_t j = 0; j < devices.size(); ++j) {
    auto shard_device = ParseDeviceString(devices[j]);
    auto shard_shape =
        CreateComputationShapeFromTensor(local_shards[j], &shard_device);
    auto populate_fn =
        [&, j, shard_device](
            const runtime::ComputationClient::TensorSource& source_tensor,
            void* dest_buffer, size_t dest_buffer_size) {
          PopulateTensorBuffer(local_shards[j], source_tensor.shape,
                               dest_buffer, dest_buffer_size, shard_device);
        };
    source_tensors.emplace_back(shard_shape, devices[j],
                                std::move(populate_fn));
  }
  return runtime::GetComputationClient()->TransferShardsToServer(
      source_tensors, GetVirtualDevice().toString(), global_shape, sharding);
}

}  // namespace torch_xla
