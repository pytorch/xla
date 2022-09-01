#include "torch_xla/csrc/xla_sharding_util.h"

#include <ATen/TensorIndexing.h>

#include <cmath>
#include <unordered_map>

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace {

using xla::internal::XlaBuilderFriend;

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

xla::HloModuleProto ShardingUtil::SpmdPartitioningPass(
    const xla::HloModuleProto& hlo_proto, bool conv_halo_exchange_always_on_lhs,
    bool choose_faster_windowed_einsum_over_mem, bool unroll_windowed_einsum,
    bool bidirectional_windowed_einsum) {
  // TODO(yeounoh) read available devices
  int64_t num_replicas = 1;
  int64_t num_partitions = 8;

  // TODO(yeounoh) propagate this down to the PJRT client
  auto execution_options = xla::CreateDefaultExecutionOptions();
  execution_options.set_use_spmd_partitioning(true);
  execution_options.set_num_replicas(num_replicas);
  execution_options.set_num_partitions(num_partitions);
  auto module_config = xla::HloModule::CreateModuleConfigFromProto(
                           hlo_proto, xla::DebugOptions(), &execution_options)
                           .ValueOrDie();
  auto module = xla::HloModule::CreateFromProto(hlo_proto, module_config,
                                                /*prohibit_empty_literal=*/true)
                    .ValueOrDie();

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
      /*allow_spmd_sharding_propagation_to_output=*/true);
  pass.AddPass<xla::spmd::SpmdPartitioner>(
      /*num_partitions=*/num_partitions,
      /*num_replicas=*/num_replicas, options,
      xla::spmd::GetDefaultCollectiveOpsCreator(
          /*num_partitions=*/num_partitions,
          /*num_replicas=*/num_replicas));
  pass.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                 /*allow_mixed_precision=*/false);
  pass.Run(module.get());

  return module.get()->ToProto();
}

std::vector<std::vector<xla::ComputationClient::DataPtr>>
ShardingUtil::InputHandler(
    std::vector<xla::ComputationClient::DataPtr> arguments,
    std::vector<std::string> devices) {
  std::vector<std::vector<xla::ComputationClient::DataPtr>> arguments_by_device(
      devices.size(),
      std::vector<xla::ComputationClient::DataPtr>(arguments.size()));
  for (int64_t argument_i = 0; argument_i < arguments.size(); ++argument_i) {
    auto shards =
        xla::ComputationClient::Get()->GetDataShards(arguments[argument_i]);
    if (shards.size() > 1) {
      // Input is sharded across addressable devices
      for (auto shard : shards) {
        int64_t device_i = ParseDeviceString(shard->device()).ordinal();
        arguments_by_device[device_i][argument_i] = shard;
      }
    } else {
      // Input is replicated across addressable devices
      int64_t source_device_i =
          ParseDeviceString(shards[0]->device()).ordinal();
      arguments_by_device[source_device_i][argument_i] = shards[0];

      auto literal = std::make_shared<xla::Literal>(std::move(
          xla::ComputationClient::Get()->TransferFromServer(shards)[0]));
      std::vector<xla::ComputationClient::TensorSource> source_tensors;
      for (int64_t device_i = 0; device_i < devices.size(); ++device_i) {
        if (device_i != source_device_i) {
          auto populate_fn =
              [&](const xla::ComputationClient::TensorSource& source_tensor,
                  void* dest_buffer, size_t dest_buffer_size) {
                std::memcpy(dest_buffer, literal->untyped_data(),
                            dest_buffer_size);
              };
          source_tensors.emplace_back(
              xla::Shape(shards[0]->shape().ToProto()),
              ParseDeviceString(absl::StrFormat(":%d", device_i)).toString(),
              std::move(populate_fn));
        }
      }

      std::vector<xla::ComputationClient::DataPtr> replicated_shards =
          xla::ComputationClient::Get()->TransferToServer(source_tensors);
      auto itr = replicated_shards.begin();
      for (int64_t device_i = 0; device_i < devices.size(); ++device_i) {
        if (device_i != source_device_i) {
          arguments_by_device[device_i][argument_i] = *itr;
          ++itr;
        }
      }
      XLA_CHECK(itr == replicated_shards.end())
          << "Replicated arguments[" << argument_i << "] on "
          << shards[0]->device() << " " << replicated_shards.size()
          << " times (expected " << (devices.size() - 1) << ").";
    }
  }

  return arguments_by_device;
}

std::vector<at::Tensor> ShardingUtil::ShardTensor(
    const at::Tensor& tensor, const xla::OpSharding sharding,
    const std::vector<std::string>& devices) {
  TF_LOG(INFO) << "ShardTensor with sharding type(" << sharding.type() << ")..."
               << std::endl;
  std::vector<at::Tensor> shards(devices.size());
  if (sharding.type() == xla::OpSharding::REPLICATED) {
    std::fill_n(shards.begin(), shards.size(), tensor);
  } else if (sharding.type() == xla::OpSharding::OTHER) {
    XLA_CHECK_EQ(devices.size(), sharding.tile_assignment_devices().size())
        << "Invalid sharding tile_assignment_devices.size(): expected "
        << devices.size() << ", actual "
        << sharding.tile_assignment_devices().size();
    // TODO(yeounoh) support higher-order topology
    XLA_CHECK(sharding.tile_shape().dimensions_size() <= 2);
    XLA_CHECK(tensor.sizes().size() >= sharding.tile_shape().dimensions_size())
        << "Data dimensions should be at least as large as the tile shape "
           "dimensions.";

    auto tile_shape = sharding.tile_assignment_dimensions();
    for (int64_t core : sharding.tile_assignment_devices()) {
      at::Tensor shard;
      if (tile_shape.size() == 1) {
        int64_t x_partition = tensor.sizes()[0] / tile_shape[0] +
                              (tensor.sizes()[0] % tile_shape[0] != 0);
        shard = tensor.index(
            {at::indexing::Slice((core / tile_shape[1]) * x_partition,
                                 (core / tile_shape[1] + 1) * x_partition)});
      } else if (tile_shape.size() == 2) {
        // ceil or ratio
        int64_t x_partition = tensor.sizes()[0] / tile_shape[0] +
                              (tensor.sizes()[0] % tile_shape[0] != 0);
        int64_t y_partition = tensor.sizes()[1] / tile_shape[1] +
                              (tensor.sizes()[1] % tile_shape[1] != 0);
        shard = tensor.index(
            {at::indexing::Slice((core / tile_shape[1]) * x_partition,
                                 (core / tile_shape[1] + 1) * x_partition),
             at::indexing::Slice((core % tile_shape[1]) * y_partition,
                                 (core % tile_shape[1] + 1) * y_partition)});
      }
      shards[core] = shard.contiguous(at::MemoryFormat::Contiguous);
    }
  } else if ((sharding.type() == xla::OpSharding::MANUAL) ||
             (sharding.type() == xla::OpSharding::TUPLE)) {
    TF_LOG(ERROR) << "Unsupported OpSharidng type " << sharding.type();
  }
  return shards;
}

}  // namespace torch_xla
