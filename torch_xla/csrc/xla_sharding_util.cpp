#include "torch_xla/csrc/xla_sharding_util.h"

#include <ATen/TensorIndexing.h>

#include <cmath>
#include <unordered_map>

#include "absl/synchronization/blocking_counter.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/aten_autograd_ops.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/thread_pool.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/protobuf_util.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/xla.pb.h"
#include "xla_sharding_util.h"

namespace torch_xla {

// Macro for defining a function that will be run at static initialization time
// to define a library of operators in the namespace. Used to define a new set
// of custom operators that do not already exist in PyTorch.
TORCH_LIBRARY_FRAGMENT(xla, m) {
  m.def(
      "xla_mark_sharding_dynamo_custom_op(Tensor input, int[][] "
      "tile_assignment, int[][] group_assignment, int[][] replication_groups, "
      "int sharding_type) -> ()",
      torch::dispatch(
          c10::DispatchKey::XLA,
          TORCH_FN(torch_xla::ShardingUtil::XlaMarkShardingDynamoCustomOp)));
}

namespace {

using xla::internal::XlaBuilderFriend;

static bool use_auto_sharding = false;

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

std::vector<int64_t> ParseStringToIntVector(const std::string& str) {
  std::istringstream ss;
  ss.str(str);
  std::vector<int64_t> result;
  for (std::string s; std::getline(ss, s, ',');) {
    try {
      result.push_back(std::stoi(s));
    } catch (std::invalid_argument const& e) {
      TF_LOG(ERROR) << "Error parsing string: " << str
                    << " with an exception: " << e.what();
    }
  }
  return result;
}

}  // namespace

bool ShardingUtil::SetHloSharding(LoweringContext* lowering_ctx) {
  bool is_sharded = false;
  for (std::pair<torch::lazy::Output, xla::XlaOp> elem :
       lowering_ctx->GetEmittedOutputs()) {
    const torch::lazy::Node* node = elem.first.node;
    const XlaNode* xla_node = dynamic_cast<const XlaNode*>(node);
    xla::HloInstructionProto* instruction =
        XlaBuilderFriend::GetInstruction(elem.second);
    const std::shared_ptr<xla::OpSharding> sharding =
        xla_node->GetSharding(elem.first.index);
    if (sharding != nullptr && sharding->type() != xla::OpSharding::UNKNOWN) {
      *instruction->mutable_sharding() = *sharding;
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
    case xla::OpSharding::UNKNOWN:
      return ShardingType::UNKNOWN;
    default:
      TF_LOG(ERROR) << "Unsupported sharding type: " << sharding.type();
  }
}

bool ShardingUtil::EqualShardingSpecs(const XLATensor::ShardingSpec& a,
                                      const XLATensor::ShardingSpec& b) {
  return xla::protobuf_util::ProtobufEquals(a.sharding, b.sharding);
}

bool ShardingUtil::EqualOpShardings(const xla::OpSharding& a,
                                    const xla::OpSharding& b) {
  return xla::protobuf_util::ProtobufEquals(a, b);
}

xla::OpSharding ShardingUtil::CreateOpSharding(
    const py::list& tile_assignment, const py::list& group_assignment,
    const py::list& replication_groups, ShardingType sharding_type) {
  TORCH_LAZY_COUNTER("CreateOpSharding", 1);
  xla::OpSharding sharding;
  switch (sharding_type) {
    case ShardingType::MANUAL: {
      sharding = xla::HloSharding::Manual().ToProto();
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
      xla::Array<int64_t> group_tile = TileListToArray(group_assignment);
      auto group_members = ExtractGroupMembers(replication_groups);
      std::vector<absl::Span<const int64_t>> group_members_view;
      group_members_view.reserve(group_members.size());
      for (auto& group : group_members) {
        auto group_view = absl::MakeConstSpan(group);
        group_members_view.push_back(group_view);
      }
      XLA_CHECK(group_tile.num_elements() == group_members_view.size());
      // The original PartialTile API is deleted in
      // https://github.com/openxla/xla/commit/728e13fb733dba2e633bdac3af6d133aa419d545.
      // Port the logic in deleted API here.
      std::vector<int64_t> new_tile_dims(group_tile.dimensions().begin(),
                                         group_tile.dimensions().end());
      new_tile_dims.push_back(group_members_view[0].size());
      auto new_tile_assignment = xla::Array<int64_t>(new_tile_dims);
      new_tile_assignment.Each(
          [&](absl::Span<const int64_t> indices, int64_t* device) {
            std::vector<int64_t> group_index(indices.begin(), indices.end());
            group_index.pop_back();
            int64_t group = group_tile(group_index);
            *device = group_members_view[group][indices.back()];
          });
      sharding = xla::HloSharding::PartialTile(new_tile_assignment).ToProto();
      break;
    }
    case ShardingType::UNKNOWN: {
      sharding = xla::HloSharding::Unknown().ToProto();
    }
    default: {
      TF_LOG(ERROR) << "Invalid arguments: sharding_type " << sharding_type;
    }
  }
  return sharding;
}

std::vector<int64_t> ShardingUtil::GetShardShape(
    const XLATensor::ShardingSpecPtr shardings) {
  auto sharding = shardings->sharding;
  auto global_shape = shardings->shape.dimensions();
  if (sharding.type() == xla::OpSharding::REPLICATED ||
      sharding.type() == xla::OpSharding::UNKNOWN) {
    std::vector<int64_t> globalShape;
    globalShape.assign(global_shape.begin(), global_shape.end());
    return globalShape;
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    auto tile_shape = sharding.tile_assignment_dimensions();
    // `shard_shape[j]` is the size of dimension `j` in the resulting shard.
    std::vector<int64_t> shard_shape;
    for (int j = 0; j < tile_shape.size(); j++) {
      if (sharding.replicate_on_last_tile_dim() && j == tile_shape.size() - 1) {
        continue;
      }
      shard_shape.push_back(global_shape[j] / tile_shape[j] +
                            (global_shape[j] % tile_shape[j] != 0));
    }

    return shard_shape;
  } else {
    XLA_CHECK(false) << "Unsupported OpSharding type " << sharding.type();
  }
}

std::vector<std::vector<at::indexing::TensorIndex>>
ShardingUtil::GetShardIndicesForMinibatchTensor(
    const std::vector<int64_t>& shard_shape,
    const std::vector<std::string>& devices) {
  std::vector<std::vector<at::indexing::TensorIndex>> shard_indices(
      devices.size());
  for (int i = 0; i < devices.size(); i++) {
    std::vector<at::indexing::TensorIndex> indices;
    // For batch dimension sharding we just change shard indices on first axis
    // and copy all indices for all remaining axes.
    for (int j = 0; j < shard_shape.size(); j++) {
      indices.push_back(at::indexing::Slice(0, shard_shape[j]));
    }
    // As the tensor is batch sharded we just care about the first dimension
    // to calculate shard indices.
    indices[0] =
        at::indexing::Slice(i * shard_shape[0], (i + 1) * shard_shape[0]);
    shard_indices[i] = indices;
  }
  return shard_indices;
}

std::vector<std::pair<int, std::vector<at::indexing::TensorIndex>>>
ShardingUtil::GetShardReplicaAndIndicesForDevices(
    const std::vector<int64_t>& shard_shape,
    const std::vector<int64_t>& tensor_shape, const xla::OpSharding sharding,
    const std::vector<std::string>& devices) {
  using namespace at::indexing;

  // `shard_indices[dev][dim]` represents the index slice for dimension `dim`
  // that belongs on device `devices[dev]` if the tensor is sharded. If
  // `sharding` is REPLICATED, `shard_indices[dev]` will only have a single
  // Ellipsis element to indicate that the tensor is replicated across all
  // dimensions.
  std::vector<std::pair<int, std::vector<TensorIndex>>> shard_indices(
      devices.size());
  auto tile_shape = sharding.tile_assignment_dimensions();
  if (sharding.type() == xla::OpSharding::REPLICATED ||
      sharding.type() == xla::OpSharding::UNKNOWN) {
    // Use Ellipsis to indicate all dimensions are replicated
    auto ellipsis = TensorIndex(Ellipsis);
    auto indices = std::vector<TensorIndex>({ellipsis});
    for (int i = 0; i < devices.size(); ++i) {
      int global_ordinal = ParseDeviceString(devices[i]).ordinal();
      shard_indices[i] = std::make_pair(global_ordinal, indices);
    }
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    auto device_index = build_index_map(devices);
    std::vector<int64_t> tile_assignment_devices(
        sharding.tile_assignment_devices().begin(),
        sharding.tile_assignment_devices().end());
    if (!sharding.iota_reshape_dims().empty()) {
      auto tileAssignment = xla::TileAssignment(
          sharding.tile_assignment_dimensions(), sharding.iota_reshape_dims(),
          sharding.iota_transpose_perm());
      tile_assignment_devices = std::vector<int64_t>(
          tileAssignment.array().begin(), tileAssignment.array().end());
    }
    for (size_t i = 0; i < tile_assignment_devices.size(); i++) {
      int64_t core = tile_assignment_devices[i];
      if (device_index.find(core) == device_index.end()) {
        // Skip any shards whose device is not part of the `devices` list.
        continue;
      }

      // The replica id for this shard. This value is only updated from 0 if
      // the sharding is partially replicated.
      int replica_id = 0;

      // Given the shard's row-major index `i`, we need to calculate shard's
      // coordinates (n_0, ..., n_d) in the tiling to generate the index
      // slices. Using `N_j = tile_shape[j]` and `0 <= n_j < N_j`, the
      // following equation needs to be solved for all n_j:
      //            `i = n_d + N_d * (n_{d-1} + N_{d-1} * (... + (N_1 *
      //            n_0)))`
      // Let `offset_j = n_j + N_j * (n_{j-1} + N_{j-1} * (... + (N_1 *
      // n_0)))`. Then `offset_d = i`, `n_j = offset_j % N_j`, and
      // `offset_{j-1} = offset_j / N_j`.
      int offset = i;
      std::vector<TensorIndex> indices;
      for (int j = tile_shape.size() - 1; j >= 0; j--) {
        int64_t n_j = offset % tile_shape[j];
        if (sharding.replicate_on_last_tile_dim() &&
            j == tile_shape.size() - 1) {
          // the last tile assignment dimension is replicated, which implies
          // that the consecutive `tile_shape[j]` devices hold the replicated.
          replica_id = n_j;
          offset /= tile_shape[j];
          continue;
        }
        // Clamp the slice bounds to the tensor shape to accurately reflect
        // the shard size without padding.
        int start = std::min(n_j * shard_shape[j], tensor_shape[j]);
        int end = std::min((n_j + 1) * shard_shape[j], tensor_shape[j]);
        auto slice = Slice(start, end);
        indices.push_back(TensorIndex(slice));
        offset /= tile_shape[j];
      }
      std::reverse(indices.begin(), indices.end());
      shard_indices[device_index[core]] = std::make_pair(replica_id, indices);
    }
  } else {
    XLA_CHECK(false) << "Unsupported OpSharding type " << sharding.type();
  }
  return shard_indices;
}

std::vector<at::Tensor> ShardingUtil::ShardTensor(
    const at::Tensor& tensor, const XLATensor::ShardingSpecPtr shardings,
    const std::vector<std::string>& devices, bool padded) {
  xla::OpSharding sharding;
  bool minibatch = false;
  if (shardings != nullptr) {
    sharding = shardings->sharding;
    minibatch = shardings->minibatch;
  }
  TF_VLOG(5) << "ShardTensor with sharding type(" << sharding.type()
             << ")... and minibatch = " << minibatch << std::endl;
  auto device_index = build_index_map(devices);
  std::vector<at::Tensor> shards(devices.size());
  if (shardings == nullptr || sharding.type() == xla::OpSharding::REPLICATED ||
      sharding.type() == xla::OpSharding::UNKNOWN) {
    std::fill_n(shards.begin(), shards.size(), tensor);
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    XLA_CHECK(sharding.tile_shape().dimensions_size() <= 2);
    XLA_CHECK(tensor.sizes().size() >= sharding.tile_shape().dimensions_size());

    auto shard_shape = GetShardShape(shardings);

    std::vector<std::vector<at::indexing::TensorIndex>> shard_indices;
    if (minibatch) {
      shard_indices = GetShardIndicesForMinibatchTensor(shard_shape, devices);
    } else {
      auto replica_and_indices = GetShardReplicaAndIndicesForDevices(
          shard_shape, tensor.sizes().vec(), sharding, devices);
      // Extract only the indices, the replica_id is unnecessary for sharding.
      std::transform(replica_and_indices.begin(), replica_and_indices.end(),
                     std::back_inserter(shard_indices),
                     [](auto& pair) { return pair.second; });
    }

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
  } else {
    XLA_CHECK(false) << "Unsupported OpSharding type " << sharding.type();
  }
  return shards;
}

std::vector<XLATensor::ShardingSpecPtr> ShardingUtil::GetOutputSharding(
    const std::vector<xla::Shape>& output_shapes,
    runtime::ComputationClient::ComputationPtr computation) {
  const auto& computation_proto = computation->computation().proto();
  size_t num_outputs = output_shapes.size();
  std::vector<xla::OpSharding> output_shardings;
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs(num_outputs);
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
    output_shardings.resize(num_outputs);
  }

  for (int i = 0; i < num_outputs; ++i) {
    sharding_specs[i] = std::make_shared<XLATensor::ShardingSpec>(
        output_shardings[i],
        MakeShapeWithDeviceLayout(output_shapes[i], XlaDeviceType::SPMD));
  }
  return sharding_specs;
}

std::vector<torch::lazy::BackendDataPtr> ShardingUtil::CreateShardedPlaceholder(
    const std::vector<XLATensor::ShardingSpecPtr>& sharding_specs) {
  std::vector<torch::lazy::BackendDataPtr> placeholders;
  placeholders.reserve(sharding_specs.size());
  for (int i = 0; i < sharding_specs.size(); ++i) {
    // Create sharded data placeholder, this will be used to
    // hold the corresponding computation results for both sharding &
    // replication.
    auto sharded_data_placeholder =
        runtime::GetComputationClient()->CreateDataPlaceholder(
            GetVirtualDevice().toString(), sharding_specs[i]->shape,
            sharding_specs[i]->sharding);

    // Register the sharded data placeholder to the tensor and its node.
    placeholders.push_back(sharded_data_placeholder);
  }
  return placeholders;
}

void ShardingUtil::PrepareOutputShardingPropagation(
    std::vector<XLATensorPtr>* tensors, absl::Span<const size_t> indices,
    runtime::ComputationClient::ComputationPtr computation,
    std::vector<torch::lazy::BackendDataPtr>* data_placeholders,
    std::vector<XLATensor::ShardingSpecPtr>* sharding_specs) {
  // Resizes the containers to `indices.size()`.
  data_placeholders->resize(indices.size());
  sharding_specs->resize(indices.size());

  std::vector<xla::Shape> output_shapes;
  output_shapes.reserve(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    auto xtensor = (*tensors)[indices[i]];
    output_shapes.push_back(xtensor->shape().get());
  }
  auto new_sharding_specs = GetOutputSharding(output_shapes, computation);
  XLA_CHECK(indices.size() == new_sharding_specs.size())
      << "Expected size: " << indices.size()
      << ", actual size: " << new_sharding_specs.size();

  for (int i = 0; i < indices.size(); ++i) {
    auto xtensor = (*tensors)[indices[i]];
    (*sharding_specs)[i] = new_sharding_specs[i];
    // Allow overwriting the sharding specs, since output sharding propagation
    // happens after any resharding that might have already taken place during
    // auto-sharding pass.
    xtensor->SetShardingSpec(*(*sharding_specs)[i], /*allow_overwrite=*/true);

    // Create sharded data placeholder, this will be used to
    // hold the corresponding computation results for both sharding &
    // replication.
    auto sharded_data_placeholder =
        runtime::GetComputationClient()->CreateDataPlaceholder(
            GetVirtualDevice().toString(), (*sharding_specs)[i]->shape,
            (*sharding_specs)[i]->sharding);

    // Register the sharded data placeholder to the tensor and its node.
    (*data_placeholders)[i] = sharded_data_placeholder;
    xtensor->data()->handle = (*data_placeholders)[i];
    // TODO(JackCaoG): Invesgate why output tensor has IR value here.
    if (xtensor->CurrentIrValue()) {
      xtensor->AssignIrValue(torch::lazy::Value());
    }
  }
}

runtime::ComputationClient::DataPtr ShardingUtil::CreateShardedData(
    const std::vector<at::Tensor>& local_shards,
    const std::vector<std::string>& devices,
    const XLATensor::ShardingSpecPtr& sharding_spec) {
  XLA_CHECK(local_shards.size() == devices.size())
      << "A device must be speficied for each shard";
  std::vector<std::shared_ptr<const runtime::TensorSource>> source_tensors;
  xla::Shape global_shape;
  xla::OpSharding sharding;
  if (sharding_spec == nullptr) {
    // Unknown type is used to mark implicitly replicated data for
    // auto-sharding.
    // TODO(yeounoh) see if we can completely rely on Unknown without inference
    // performance degradation.
    sharding = ShardingUtil::GetAutoSharding()
                   ? xla::HloSharding::Unknown().ToProto()
                   : xla::HloSharding::Replicate().ToProto();
    // if replicated, global_shape is shape of the tensor.
    auto first_device = ParseDeviceString(devices[0]);
    global_shape =
        CreateComputationShapeFromTensor(local_shards[0], &first_device);
  } else {
    global_shape = sharding_spec->shape;
    sharding = sharding_spec->sharding;
  }
  for (int64_t j = 0; j < devices.size(); ++j) {
    auto shard_device = ParseDeviceString(devices[j]);
    auto shard_shape =
        CreateComputationShapeFromTensor(local_shards[j], &shard_device);
    source_tensors.push_back(std::make_shared<runtime::AtenSource>(
        local_shards[j], shard_shape, devices[j]));
  }
  return runtime::GetComputationClient()->TransferShardsToDevice(
      source_tensors, GetVirtualDevice().toString(), global_shape, sharding);
}

std::vector<int64_t> ShardingUtil::GetAutoShardingMesh() {
  // Auto-sharding uses mesh_shape = {n_devices, 1} if XLA_AUTO_SPMD_MESH
  // is not set. XLA_AUTO_SPMD_MESH takes a form of string, "2,2" which
  // corresponds to a 2-by-2 mesh.
  std::vector<int64_t> mesh_shape = ParseStringToIntVector(
      runtime::sys_util::GetEnvString("XLA_AUTO_SPMD_MESH", ""));
  if (!mesh_shape.empty()) {
    int64_t total_devices = 1;
    for (auto i : mesh_shape) {
      total_devices *= i;
    }
    XLA_CHECK_EQ(total_devices,
                 runtime::GetComputationClient()->GetAllDevices().size())
        << "Invalid auto-sharding mesh_shape: "
        << absl::StrJoin(mesh_shape, ",");
  }
  return mesh_shape;
}

std::vector<int64_t> ShardingUtil::GetAutoShardingMeshIds(
    const xla::HloModuleProto& module) {
  // Return the first non-default (iota) mesh ids arrangement, as we expect
  // only one such assignment and/or the logical mesh device assignment should
  // be compatible with the other arrangements in the HLO. This is a work-around
  // as the auto-sharding pass takes only one arrangement for now.
  // TODO(yeounoh) this was not necessary before; replace if this can be done
  // during the auto-sharding pass.
  int64_t n_devices = runtime::GetComputationClient()->GetAllDevices().size();
  std::vector<int64_t> device_mesh_ids = std::vector<int64_t>(n_devices);
  std::iota(device_mesh_ids.begin(), device_mesh_ids.end(), 0);

  // Unforuntately, we have to go through the instructions since
  // `spmd_parameters_shardings` is not available.
  for (auto computation : module.computations()) {
    for (auto instruction : computation.instructions()) {
      if (instruction.opcode() == "parameter" && instruction.has_sharding()) {
        xla::OpSharding sharding = instruction.sharding();
        auto tile_assignment_devices = sharding.tile_assignment_devices();
        if (!tile_assignment_devices.empty()) {
          auto new_mesh_ids = std::vector<int64_t>(
              tile_assignment_devices.begin(), tile_assignment_devices.end());
          // return the first non-default (iota) device assigments.
          if (new_mesh_ids != device_mesh_ids) {
            return new_mesh_ids;
          }
        }
      }
    }
  }
  // return the default (iota) device assignments.
  return device_mesh_ids;
}

void ShardingUtil::ReshardParameters(
    const xla::HloModuleProto& module, std::vector<XLATensorPtr>* tensors,
    std::vector<torch::lazy::BackendDataPtr>* parameters,
    std::vector<const torch::lazy::Node*>* nodes) {
  // Extract input shardings generated from auto-sharding pass.
  std::vector<xla::OpSharding> input_shardings;
  if (module.spmd_parameters_shardings().size() == 1 &&
      module.spmd_parameters_shardings()[0].type() == xla::OpSharding::TUPLE) {
    auto tuple_shardings =
        module.spmd_parameters_shardings()[0].tuple_shardings();
    input_shardings = std::vector<xla::OpSharding>(tuple_shardings.begin(),
                                                   tuple_shardings.end());
  } else {
    for (auto sharding : module.spmd_parameters_shardings()) {
      input_shardings.push_back(sharding);
    }
  }
  if (input_shardings.size() == 0) {
    TF_VLOG(3) << "ReshardParamters... skip with empty input_shardings.";
    return;
  }
  XLA_CHECK_EQ(input_shardings.size(), parameters->size());

  // Reshard parameters as needed, as with a new sharding spec.
  std::vector<runtime::ComputationClient::DataPtr> data =
      UnwrapXlaData(*parameters);

  std::vector<size_t> reshard_indices;
  std::vector<runtime::ComputationClient::DataPtr> data_to_reshard;
  std::vector<xla::OpSharding> shardings_to_reshard;
  for (int i = 0; i < input_shardings.size(); ++i) {
    XLA_CHECK(input_shardings[i].type() != xla::OpSharding::UNKNOWN)
        << "Resharding by UNKNOWN sharding type is not allowed.";
    // Skip re-sharding if not necessary.
    if (!xla::protobuf_util::ProtobufEquals(data[i]->GetSharding(),
                                            input_shardings[i])) {
      reshard_indices.push_back(i);
      data_to_reshard.push_back(data[i]);
      shardings_to_reshard.push_back(input_shardings[i]);
    }
  }
  if (reshard_indices.size() == 0) {
    TF_VLOG(3) << "ReshardParamters... skip with no new shardings.";
    return;
  }
  TF_VLOG(3) << "ReshardParamters... resharding " << reshard_indices.size()
             << " parameters.";

  TORCH_LAZY_COUNTER("ReshardParameters", 1);

  // Construct parameter handle to XlaNode mappping for faster look-up.
  std::unordered_map<torch::lazy::BackendData::Handle, const torch::lazy::Node*>
      xla_node_map;
  for (const torch::lazy::Node* node : *nodes) {
    const auto backend_data =
        torch::lazy::getBackend()->GetComputationDataFromNode(node);
    if (backend_data) {
      torch::lazy::BackendData::Handle handle = backend_data->GetHandle();
      xla_node_map[handle] = node;
    }
  }

  std::vector<torch::lazy::BackendDataPtr> outputs;
  outputs.reserve(reshard_indices.size());
  // Groupping is computationally more efficient but increases memory
  // consumption. It is groupped by default, but can be overriden for
  // more-granular control over the peak memory consumption.
  bool group_sharding =
      runtime::sys_util::GetEnvBool("XLA_AUTO_USE_GROUP_SHARDING", true);
  if (group_sharding) {
    outputs = WrapXlaData(runtime::GetComputationClient()->ReshardData(
        data_to_reshard, shardings_to_reshard));
  } else {
    for (int i = 0; i < data_to_reshard.size(); ++i) {
      auto output = WrapXlaData(runtime::GetComputationClient()->ReshardData(
          {data_to_reshard[i]}, {shardings_to_reshard[i]}));
      outputs.insert(outputs.end(), output.begin(), output.end());
    }
  }
  XLA_CHECK_EQ(outputs.size(), reshard_indices.size());

  for (int i = 0; i < outputs.size(); ++i) {
    (*parameters)[reshard_indices[i]] = outputs[i];
    auto it_node = xla_node_map.find(data_to_reshard[i]->GetHandle());
    XLA_CHECK(it_node != xla_node_map.end())
        << "xla_node_map does not contain " << data_to_reshard[i]->ToString()
        << ", target sharding: " << shardings_to_reshard[i].DebugString();
    auto device_data_node = DeviceData::Cast(it_node->second);
    device_data_node->SetSharding(shardings_to_reshard[i], 0);
  }
}

void ShardingUtil::XlaMarkSharding(const at::Tensor& input,
                                   xla::OpSharding sharding) {
  TORCH_LAZY_COUNTER("XlaMarkSharding", 1);
  XLA_CHECK(UseVirtualDevice())
      << "Please enable SPMD via `torch_xla.runtime.use_spmd()`";
  XLA_CHECK(sharding.type() != xla::OpSharding::UNKNOWN)
      << "Can't explicilty annotate with UNKNOWN sharding type.";
  XLATensorPtr xtensor = bridge::GetXlaTensor(input);
  XLATensor::ShardingSpecPtr new_sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(
          sharding, MakeShapeWithDeviceLayout(
                        xtensor->shape(), static_cast<XlaDeviceType>(
                                              xtensor->GetDevice().type())));

  // For Non DeviceData IR values, we directly attach the sharding spec
  // to the xtensor.
  const DeviceData* device_data_node = nullptr;
  if (xtensor->CurrentIrValue()) {
    device_data_node = DeviceData::Cast(xtensor->CurrentIrValue().node.get());
    if (!device_data_node) {
      tensor_methods::custom_sharding_(xtensor, new_sharding_spec);
      return;
    }
  }

  // For data, we need to deal with the data transfers between
  // host and device.
  at::Tensor cpu_tensor;
  if (xtensor->CurrentTensorData().has_value()) {
    TORCH_LAZY_COUNTER("VirtualDeviceUsage", 1);
    // When virtual device is enabled for SPMD, we defer the initial
    // data transfer to the device and retain the original data on the
    // host, until the sharded data transfer.
    cpu_tensor = xtensor->CurrentTensorData().value();
  } else {
    // A new input tensor is not expected to be sharded. But sometimes,
    // the same input is called for sharding annotation over multiple steps,
    // in which case we can skip if it's the same sharding; however, if it's
    // the same input with a different sharding then we block & ask the user
    // to clear the existing sharding first.
    XLATensor::ShardingSpecPtr current_sharding_spec = xtensor->sharding_spec();
    if (current_sharding_spec) {
      if (ShardingUtil::EqualShardingSpecs(*new_sharding_spec,
                                           *current_sharding_spec)) {
        return;
      }
      auto type = current_sharding_spec->sharding.type();
      if (type != xla::OpSharding::REPLICATED &&
          type != xla::OpSharding::UNKNOWN) {
        XLA_CHECK(false) << "Existing annotation must be cleared first: "
                         << current_sharding_spec->sharding.DebugString();
      }
    }

    // If the at::Tensor data is not present, we need to re-download the
    // tensor from the physical device to CPU. In that case, the value
    // must be present on the backend device.
    XLA_CHECK((xtensor->CurrentDataHandle() &&
               xtensor->CurrentDataHandle()->HasValue()) ||
              device_data_node != nullptr)
        << "Cannot shard tensor. Data does not present on any device.";
    std::vector<XLATensorPtr> xla_tensors{xtensor};
    cpu_tensor = XLAGraphExecutor::Get()->GetTensors(&xla_tensors)[0];
  }
  auto xla_data = CreateTensorsData(
      std::vector<at::Tensor>{cpu_tensor},
      std::vector<XLATensor::ShardingSpecPtr>{new_sharding_spec},
      std::vector<std::string>{GetVirtualDevice().toString()})[0];
  xtensor->SetXlaData(xla_data);
  xtensor->SetShardingSpec(*new_sharding_spec);

  // Register sharded tensor data.
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
}

void ShardingUtil::XlaMarkShardingDynamoCustomOp(
    const at::Tensor& input, c10::List<at::IntArrayRef> tile_assignment,
    c10::List<at::IntArrayRef> group_assignment,
    c10::List<at::IntArrayRef> replication_groups, int64_t sharding_type) {
  py::list tile_assignment_py = py::list();
  for (int i = 0; i < tile_assignment.size(); i++) {
    py::list pylist = py::list();
    for (int64_t t : tile_assignment[i].get().toIntList()) {
      pylist.append(t);
    }
    tile_assignment_py.append(pylist);
  }

  py::list group_assignment_py = py::list();
  for (int i = 0; i < group_assignment.size(); i++) {
    py::list pylist = py::list();
    for (int64_t t : group_assignment[i].get().toIntList()) {
      pylist.append(t);
    }
    group_assignment_py.append(pylist);
  }

  py::list replication_groups_py = py::list();
  for (int i = 0; i < replication_groups.size(); i++) {
    py::list pylist = py::list();
    for (int64_t t : replication_groups[i].get().toIntList()) {
      pylist.append(t);
    }
    replication_groups_py.append(pylist);
  }

  xla::OpSharding op_sharding = ShardingUtil::CreateOpSharding(
      tile_assignment_py, group_assignment_py, replication_groups_py,
      ShardingUtil::ShardingType(sharding_type));

  ShardingUtil::XlaMarkSharding(input, op_sharding);
}

void ShardingUtil::SetAutoSharding() {
  // This stays on throughout the program.
  use_auto_sharding = true;
}
bool ShardingUtil::GetAutoSharding() {
  if (runtime::sys_util::GetEnvBool("XLA_AUTO_SPMD", false)) {
    use_auto_sharding = true;
  }
  return use_auto_sharding;
}
}  // namespace torch_xla
