
#include "torch_xla/csrc/xla_sharding_util.h"

#include <unordered_map>

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace torch_xla {
namespace {

using xla::internal::XlaBuilderFriend;

}  // namespace

void ShardingUtil::SetHloSharding(LoweringContext* lowering_ctx) {
  for (std::pair<torch::lazy::Output, xla::XlaOp> elem :
       lowering_ctx->GetEmittedOutputs()) {
    const torch::lazy::Node* node = elem.first.node;
    const XlaNode* xla_node = dynamic_cast<const XlaNode*>(node);
    auto instruction = XlaBuilderFriend::GetInstruction(elem.second);
    if (xla_node->GetSharding() != nullptr) {
      //*instruction->mutable_sharding() = *xla_node->GetSharding();
      {
        // Annotate the full-shape input with the sharding.
        xla::XlaScopedShardingAssignment assign_sharding(
            lowering_ctx->builder(), *xla_node->GetSharding());
        xla::CustomCall(lowering_ctx->builder(),
                        /*call_target_name=*/"Sharding",
                        xla_node->Lower(lowering_ctx), xla_node->xla_shape(),
                        /*opaque=*/"");
      }
    }
  }
}

// This is called separately before xrt compilation. This is also useful
// for debugging partitioned HLO computation and sharding propation.
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

  auto collective_ops_creator = xla::spmd::GetDefaultCollectiveOpsCreator(
      /*num_partitions=*/num_partitions, /*num_replicas=*/num_replicas);

  xla::HloPassPipeline pass("spmd-partitioning");
  pass.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                 /*allow_mixed_precision=*/false);
  pass.AddPass<xla::ShardingPropagation>(/*is_spmd=*/true);
  pass.AddPass<xla::spmd::SpmdPartitioner>(/*num_partitions=*/num_partitions,
                                           /*num_replicas=*/num_replicas,
                                           options, collective_ops_creator);
  pass.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                 /*allow_mixed_precision=*/false);
  pass.Run(module.get());

  return module.get()->ToProto();
}

}  // namespace torch_xla
