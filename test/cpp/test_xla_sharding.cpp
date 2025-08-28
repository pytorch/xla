#include <ATen/ATen.h>
#include <google/protobuf/repeated_field.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <iostream>

#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/status.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_sharding_util.h"
#include "xla/protobuf_util.h"
#include "xla/xla_data.pb.h"

namespace torch_xla {
namespace cpp_test {
namespace {
bool XlaDataValuesEqual(torch::lazy::BackendDataPtr a,
                        torch::lazy::BackendDataPtr b,
                        at::ScalarType element_type) {
  XLA_ASSIGN_OR_THROW(std::vector<at::Tensor> tensors,
                      XlaDataToTensors({a, b}, {element_type, element_type}));
  return TensorCompare(tensors[0], tensors[1]);
}
}  // namespace

class XLAShardingTest : public AtenXlaTensorTestBase {
 protected:
  static void SetUpTestCase() {
    setenv("XLA_USE_SPMD", "1", /*overwrite=*/true);
    CommonSetup();
  }
};

TEST_F(XLAShardingTest, GetShardShapeTiled) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);

  auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  // For tiled sharding, each dimension should be halved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({4, 4}));
}

TEST_F(XLAShardingTest, GetShardShapeReplicated) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3};

  auto xla_sharding = xla::HloSharding::Replicate().ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);

  auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  // For replicated sharding, each dimension should be preserved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({8, 7}));
}

TEST_F(XLAShardingTest, GetShardIndicesForDevicesTiled) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3"};

  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  auto replica_and_indices = ShardingUtil::GetShardReplicaAndIndicesForDevices(
      shard_shape, tensor.sizes().vec(), sharding, devices);
  EXPECT_EQ(replica_and_indices.size(), devices.size());
  /* Tiled indices should be:
                 dim=0 dim=1
       device=0  [0:4,  0:4]
       device=1  [0:4,  4:7]
       device=2  [4:8,  0:4]
       device=3  [4:8,  4:7] */
  std::vector<std::vector<int>> slice_starts = {{0, 0}, {0, 4}, {4, 0}, {4, 4}};
  std::vector<std::vector<int>> slice_ends = {{4, 4}, {4, 7}, {8, 4}, {8, 7}};
  for (int device = 0; device < replica_and_indices.size(); ++device) {
    auto& shard_replica_id = replica_and_indices[device].first;
    EXPECT_EQ(shard_replica_id,
              0);  // Shard replica_id is always 0 for tiled sharding.
    auto& shard_indices = replica_and_indices[device].second;
    EXPECT_EQ(shard_indices.size(), tensor.sizes().size());
    for (int dim = 0; dim < shard_indices.size(); ++dim) {
      EXPECT_TRUE(shard_indices[dim].is_slice());
      auto slice = shard_indices[dim].slice();
      EXPECT_EQ(slice.start(), slice_starts[device][dim]);
      EXPECT_EQ(slice.stop(), slice_ends[device][dim]);
      EXPECT_EQ(slice.step(), 1);
    }
  }
}

TEST_F(XLAShardingTest, GetShardIndicesForDevicesReplicated) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3"};

  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3};

  auto xla_sharding = xla::HloSharding::Replicate().ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  auto replica_and_indices = ShardingUtil::GetShardReplicaAndIndicesForDevices(
      shard_shape, tensor.sizes().vec(), sharding, devices);
  EXPECT_EQ(replica_and_indices.size(), devices.size());
  for (int i = 0; i < devices.size(); ++i) {
    auto& replica_id = replica_and_indices[i].first;
    EXPECT_EQ(replica_id, i);  // Shard replica_id should equal global ordinal.
    auto& shard_indices = replica_and_indices[i].second;
    EXPECT_EQ(shard_indices.size(), 1);
    EXPECT_TRUE(shard_indices[0].is_ellipsis());
  }
}

TEST_F(XLAShardingTest, ShardTensor1D) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 1D tiled
  at::Tensor tensor = at::ones({8}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::OpSharding xla_sharding =
      xla::HloSharding::Tile1D(
          CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice()),
          devices.size())
          .ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1}));
  EXPECT_EQ(shards[1].sizes(), c10::ArrayRef<long>({1}));
}

TEST_F(XLAShardingTest, ShardTensor2D) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 2D tiled, The first dim is halved and the last replicated. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1, 2, 3},
      {4, 5, 6, 7},
  });
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({4, 1, 4}));
}

TEST_F(XLAShardingTest, ShardTensor3D) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 3D tiled, the first dim is replicated and the last halved. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array3D<int64_t> cube({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}});
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(cube).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 1, 2}));
}

TEST_F(XLAShardingTest, ShardTensorReplicated) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // Replicated, all shards should be identical.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::OpSharding xla_sharding = xla::HloSharding::Replicate().ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 7, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 7, 4}));
}

TEST_F(XLAShardingTest, ShardTensor4D) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 4D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  at::Tensor tensor = at::ones({1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array4D<int64_t> tesseract({{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}});
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(tesseract).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 1, 2}));
}

TEST_F(XLAShardingTest, ShardTensor4DPadded) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 4D tiled and padded, all shard sizes should be identical.
  at::Tensor tensor = at::ones({1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array4D<int64_t> tesseract({{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}});
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(tesseract).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
}

TEST_F(XLAShardingTest, ShardTensor5D) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 5D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  at::Tensor tensor = at::ones({10, 1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array<int64_t> hypercube(std::vector<int64_t>{1, 1, 2, 2, 2});
  hypercube.FillIota(0);
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(hypercube).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 3, 2}));
}

TEST_F(XLAShardingTest, ShardTensor5DPadded) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // 5D tiled and padded, all shard sizes should be identical.
  at::Tensor tensor = at::ones({10, 1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array<int64_t> hypercube(std::vector<int64_t>{1, 1, 2, 2, 2});
  hypercube.FillIota(0);
  xla::OpSharding xla_sharding = xla::HloSharding::Tile(hypercube).ToProto();
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
}

TEST_F(XLAShardingTest, ShardTensorMultiHostStartOfMesh) {
  std::vector<std::string> devices = {"TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // 2D tiled, The first dim is halved and the last replicated.
  // For devices at the start of the mesh, all shards should have the same
  // unpadded shape.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {4, 5, 0, 1},
      {6, 7, 2, 3},
  });
  auto xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {4, 5, 0, 1, 6, 7, 2, 3};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({4, 2, 4}));
}

TEST_F(XLAShardingTest, ShardTensorMultiHostEndOfMesh) {
  std::vector<std::string> devices = {"TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // When this host's devices are at the end of the mesh, the last shard should
  // be smaller in dim=2 because it's not evenly divisible.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1, 4, 5},
      {2, 3, 6, 7},
  });
  auto xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 4, 5, 2, 3, 6, 7};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({4, 1, 4}));
}

TEST_F(XLAShardingTest, ShardTensorMiniBatch) {
  std::vector<std::string> devices = {"TPU:4", "TPU:5", "TPU:6", "TPU:7"};
  at::Tensor minibatch_tensor =
      at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape global_shape = CreateComputationShapeFromTensor(
      minibatch_tensor, bridge::GetDefaultDevice());
  global_shape.set_dimensions(
      0, minibatch_tensor.sizes()[0] * 2);  // Assuming 2 hosts
  xla::Array3D<int64_t> mesh({
      {{0}},
      {{1}},
      {{2}},
      {{3}},
      {{4}},
      {{5}},
      {{6}},
      {{7}},
  });

  auto xla_sharding = xla::HloSharding::Tile(mesh).ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  auto sharding_spec = std::make_shared<XLATensor::ShardingSpec>(
      sharding, global_shape, /*minibatch=*/true);
  auto shards = ShardingUtil::ShardTensor(minibatch_tensor, sharding_spec,
                                          devices, /*padded=*/true);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({2, 7, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({2, 7, 4}));
}

TEST_F(XLAShardingTest, EqualShardingSpecsSameSpecs) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  auto xla_sharding = xla::HloSharding::Tile({
                                                 {0, 1, 2, 3},
                                                 {4, 5, 6, 7},
                                             })
                          .ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  XLATensor::ShardingSpec tiled_2d(sharding, tensor_shape);

  // Test that identical sharding specs are equal
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_2d));
}

TEST_F(XLAShardingTest, EqualShardingSpecsDifferentTiledSpecs) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());

  // Create 2D tiled sharding
  auto xla_sharding_2d = xla::HloSharding::Tile({
                                                    {0, 1, 2, 3},
                                                    {4, 5, 6, 7},
                                                })
                             .ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};
  torch_xla::OpSharding sharding_2d(xla_sharding_2d,
                                    denormalized_tile_assignment);
  XLATensor::ShardingSpec tiled_2d(sharding_2d, tensor_shape);

  // Create 3D tiled sharding
  auto xla_sharding_3d =
      xla::HloSharding::Tile({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}).ToProto();
  torch_xla::OpSharding sharding_3d(xla_sharding_3d,
                                    denormalized_tile_assignment);
  XLATensor::ShardingSpec tiled_3d(sharding_3d, tensor_shape);

  // Test that different tiled sharding specs are not equal
  EXPECT_FALSE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_3d));
}

TEST_F(XLAShardingTest, EqualShardingSpecsReplicatedSpecs) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());

  auto xla_sharding = xla::HloSharding::Replicate().ToProto();
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};
  torch_xla::OpSharding sharding(xla_sharding, denormalized_tile_assignment);
  XLATensor::ShardingSpec replicated(sharding, tensor_shape);

  // Test that identical replicated sharding specs are equal
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(replicated, replicated));
}

TEST_F(XLAShardingTest, EqualShardingSpecsTiledVsReplicated) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  std::vector<int64_t> denormalized_tile_assignment = {0, 1, 2, 3, 4, 5, 6, 7};

  // Create tiled sharding
  auto xla_sharding_tiled = xla::HloSharding::Tile({
                                                       {0, 1, 2, 3},
                                                       {4, 5, 6, 7},
                                                   })
                                .ToProto();
  torch_xla::OpSharding sharding_tiled(xla_sharding_tiled,
                                       denormalized_tile_assignment);
  XLATensor::ShardingSpec tiled_2d(sharding_tiled, tensor_shape);

  // Create replicated sharding
  auto xla_sharding_replicated = xla::HloSharding::Replicate().ToProto();
  torch_xla::OpSharding sharding_replicated(xla_sharding_replicated,
                                            denormalized_tile_assignment);
  XLATensor::ShardingSpec replicated(sharding_replicated, tensor_shape);

  // Test that tiled and replicated sharding specs are not equal
  EXPECT_FALSE(ShardingUtil::EqualShardingSpecs(tiled_2d, replicated));
}

TEST_F(XLAShardingTest, CreateTensorsData) {
  if (torch_xla::runtime::sys_util::GetEnvString(
          torch_xla::runtime::env::kEnvPjRtDevice, "") == "") {
    GTEST_SKIP() << "`PJRT_DEVICE` is not set.";
  }

  std::vector<at::Tensor> tensors(3);
  auto tensor = at::ones({8, 8}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  std::fill_n(tensors.begin(), tensors.size(), tensor);
  std::vector<std::string> devices(3);
  std::fill_n(devices.begin(), devices.size(),
              bridge::GetDefaultDevice()->toString());
  auto replicate_xla_sharding = xla::HloSharding::Replicate().ToProto();
  auto unknown_xla_sharding = xla::HloSharding::Unknown().ToProto();
  torch_xla::OpSharding replicate_sharding(replicate_xla_sharding,
                                           std::nullopt);
  torch_xla::OpSharding unknown_sharding(unknown_xla_sharding, std::nullopt);
  std::vector<XLATensor::ShardingSpecPtr> shardings = {
      nullptr,
      std::make_shared<XLATensor::ShardingSpec>(replicate_sharding,
                                                tensor_shape),
      std::make_shared<XLATensor::ShardingSpec>(unknown_sharding,
                                                tensor_shape)};
  std::vector<torch::lazy::BackendDataPtr> tensors_data =
      CreateTensorsData(tensors, shardings, devices);

  int64_t n_devices =
      torch_xla::runtime::GetComputationClientOrDie()->GetLocalDevices().size();
  if (n_devices > 1) {
    // null sharding is treated as replicated.
    auto xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[0]);
    std::vector<torch_xla::runtime::ComputationClient::DataPtr> shards =
        torch_xla::runtime::GetComputationClientOrDie()->GetDataShards(
            xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(tensors_data[0], shards[0], at::kFloat));

    // Returns multiple input shards, explicitly replicated
    auto sharded_xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[1]);
    shards = torch_xla::runtime::GetComputationClientOrDie()->GetDataShards(
        sharded_xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(sharded_xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(shards[0], shards[1], at::kFloat));

    // Returns multiple input shards, implicitly replicated
    sharded_xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[2]);
    shards = torch_xla::runtime::GetComputationClientOrDie()->GetDataShards(
        sharded_xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(sharded_xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(shards[0], shards[1], at::kFloat));
  }
}

TEST_F(XLAShardingTest, PrepareOutputShardingPropagation) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4, 4});
  int64_t n_devices =
      torch_xla::runtime::GetComputationClientOrDie()->GetLocalDevices().size();
  xla::Array<int64_t> tile_assignment({1, n_devices});
  tile_assignment.FillIota(0);
  xla::OpSharding tiled = xla::HloSharding::Tile(tile_assignment).ToProto();

  // Build simple addition with a sharded input.
  xla::XlaBuilder b("builder");
  b.SetSharding(tiled);
  auto x = xla::Parameter(&b, 0, shape, "p0");
  b.ClearSharding();
  auto y = xla::Add(x, xla::ConstantR0<float>(&b, 3));
  XLA_ASSIGN_OR_THROW(xla::XlaComputation xla_computation,
                      b.Build(/*remove_dynamic_dimensions=*/false));

  std::vector<XLATensorPtr> tensors{XLATensor::Create(
      torch_xla::runtime::GetComputationClientOrDie()->CreateDataPlaceholder(
          bridge::GetDefaultDevice()->toString(), std::move(shape)))};
  std::vector<std::vector<int64_t>> denormalized_tile_assignments;
  for (auto tensor : tensors) {
    auto sharding_spec = tensor->sharding_spec();
    if (sharding_spec) {
      denormalized_tile_assignments.push_back(
          sharding_spec->sharding.GetDenormalizedTileAssignment());
    }
  }
  std::vector<torch_xla::runtime::ComputationClient::CompileInstance> instances;
  instances.push_back(
      {std::move(xla_computation),
       bridge::GetDefaultDevice()->toString(),
       {bridge::GetDefaultDevice()->toString()},
       &shape,
       /*should_wrap_parameter=*/false,
       /*is_sharded=*/true,
       /*allow_spmd_sharding_propagation_to_output=*/true,
       /*denormalized_tile_assignments=*/denormalized_tile_assignments});

  std::vector<
      std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>>
      computations = torch_xla::runtime::GetComputationClientOrDie()->Compile(
          std::move(instances));
  torch_xla::runtime::ComputationClient::ComputationPtr computation =
      std::make_shared<torch_xla::runtime::ComputationClient::Computation>(
          "add", std::move(computations[0]->move_computation()));

  // Prepare output sharding propagation, expect a sharded output placeholder.
  std::vector<torch::lazy::BackendDataPtr> data_placeholders;
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs;
  ShardingUtil::PrepareOutputShardingPropagation(
      &tensors, {0}, computation, &data_placeholders, &sharding_specs);

  // Check if the output sharding spec is correctly extracted.
  EXPECT_EQ(sharding_specs.size(), 1);
  if (n_devices > 1) {
    // Tiled sharding requires multiple devices.
    EXPECT_TRUE(xla::protobuf_util::HaveSameSerialization(
        tiled, sharding_specs[0]->sharding.GetXlaOpSharding()));
  } else {
    // Sincle device execution defaults to replication sharding.
    EXPECT_TRUE(xla::protobuf_util::HaveSameSerialization(
        xla::HloSharding::Replicate().ToProto(),
        sharding_specs[0]->sharding.GetXlaOpSharding()));
  }

  // Check if the placeholder is on a SPMD device (sharded) with no real values.
  EXPECT_EQ(data_placeholders.size(), 1);
  EXPECT_EQ(
      std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
          data_placeholders[0])
          ->device(),
      "SPMD:0");
  EXPECT_FALSE(
      std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
          data_placeholders[0])
          ->HasValue());
}

}  // namespace cpp_test
}  // namespace torch_xla
