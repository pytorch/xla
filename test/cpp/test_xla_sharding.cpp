#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <iostream>

#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {
namespace {

void RunPartitioner(
    HloModule* module,
    absl::optional<spmd::SpmdPartitionerOptions> options = absl::nullopt) {
  jellyfish::TpuSpmdPartitioner partitioner(
      /*num_partitions=*/replica_count_, /*num_replicas=*/1,
      spmd::SPMDCollectiveOpsCreator{
          [](HloComputation::Builder* b) {
            return b->AddInstruction(HloInstruction::CreateReplicaId());
          },
          [](HloComputation::Builder* b, HloInstruction* operand,
             HloComputation* reduction,
             const std::vector<std::vector<int64_t>>& partition_subgroups,
             int64_t channel_id) {
            std::vector<ReplicaGroup> groups(partition_subgroups.size());
            for (int64_t i = 0; i < groups.size(); ++i) {
              for (int64_t r : partition_subgroups[i]) {
                groups[i].add_replica_ids(r);
              }
            }
            return b->AddInstruction(HloInstruction::CreateAllReduce(
                operand->shape(), {operand}, reduction, groups,
                /*constrain_layout=*/false, absl::nullopt,
                /*use_global_device_ids=*/false));
          },
          [](HloComputation::Builder* b, HloInstruction* operand,
             std::vector<std::pair<int64_t, int64_t>>& src_dst_pairs,
             int64_t channel_id) {
            return b->AddInstruction(HloInstruction::CreateCollectivePermute(
                operand->shape(), operand, src_dst_pairs, absl::nullopt));
          },
          [](HloComputation::Builder* b,
             absl::Span<HloInstruction* const> operands,
             const std::vector<std::vector<int64_t>>& partition_subgroups,
             int64_t channel_id, absl::optional<int64_t> split_dimension) {
            std::vector<Shape> shapes(operands.size(), operands[0]->shape());
            const Shape& output_shape = (operands.size() == 1)
                                            ? operands[0]->shape()
                                            : ShapeUtil::MakeTupleShape(shapes);
            std::vector<ReplicaGroup> groups(partition_subgroups.size());
            for (int64_t i = 0; i < groups.size(); ++i) {
              for (int64_t r : partition_subgroups[i]) {
                groups[i].add_replica_ids(r);
              }
            }
            return b->AddInstruction(HloInstruction::CreateAllToAll(
                output_shape, operands, groups,
                /*constrain_layout=*/false, absl::nullopt, split_dimension));
          },
          [](HloComputation::Builder* b, HloInstruction* operand,
             const Shape& ag_shape,
             const std::vector<std::vector<int64_t>>& partition_subgroups,
             int64_t channel_id, int64_t all_gather_dimension) {
            std::vector<ReplicaGroup> device_groups(partition_subgroups.size());
            for (int64_t i = 0; i < partition_subgroups.size(); ++i) {
              for (int64_t pid : partition_subgroups[i]) {
                // The partition IDs are in fact replica IDs in this test.
                device_groups[i].add_replica_ids(pid);
              }
            }
            return b->AddInstruction(HloInstruction::CreateAllGather(
                ag_shape, {operand}, all_gather_dimension, device_groups,
                /*constrain_layout=*/false, absl::nullopt,
                /*use_global_device_ids=*/false));
          },
      },
      options);
  ASSERT_TRUE(partitioner.Run(module).ok());
  VLOG(1) << module->ToString();
}

}  // namespace

class XLAShardingTest : public AtenXlaTensorTestBase {};

TEST_F(XLAShardingTest, TestSPMDPartitioner) {
  const int64_t replica_count_ = 8;

  absl::string_view hlo_string =
      R"(
      HloModule module

      ENTRY entry {
        constant = f32[3,3]{1,0} constant({{1,3,7},{5,1,4},{1,2,8}}),
          sharding={replicated}
        constant.1 = f32[3,3]{1,0} constant({{2,7,2},{2,9,2},{2,6,2}}),
          sharding={replicated}
        multiply = f32[3,3]{1,0} multiply(constant, constant.1),
          sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
        add = f32[3,3]{1,0} add(multiply, constant.1),
          sharding={devices=[4,1]0,1,2,3}
        ROOT copy = f32[3,3]{1,0} copy(add),
          sharding={replicated}
      }
      )";
  // Run SPMDPartitioner
}

}  // namespace cpp_test
}  // namespace torch_xla
