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
  std::vector<at::Tensor> tensors =
      XlaDataToTensors({a, b}, {element_type, element_type});
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

TEST_F(XLAShardingTest, GetShardShape) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);

  auto shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  // For tiled sharding, each dimension should be halved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({4, 4}));

  sharding_spec->sharding = xla::HloSharding::Replicate().ToProto();
  shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  // For replicated sharding, each dimension should be preserved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({8, 7}));
}

TEST_F(XLAShardingTest, GetShardIndicesForDevices) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3"};

  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
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
  sharding = xla::HloSharding::Replicate().ToProto();
  sharding_spec->sharding = sharding;
  shard_shape = ShardingUtil::GetShardShape(sharding_spec);
  replica_and_indices = ShardingUtil::GetShardReplicaAndIndicesForDevices(
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

TEST_F(XLAShardingTest, ShardTensor) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // 1D tiled
  at::Tensor tensor = at::ones({8}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::OpSharding sharding =
      xla::HloSharding::Tile1D(
          CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice()),
          devices.size())
          .ToProto();
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1}));
  EXPECT_EQ(shards[1].sizes(), c10::ArrayRef<long>({1}));

  // 2D tiled, The first dim is halved and the last replicated. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1, 2, 3},
      {4, 5, 6, 7},
  });
  sharding = xla::HloSharding::Tile(mesh).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({4, 1, 4}));

  // 3D tiled, the first dim is replicated and the last halved. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  xla::Array3D<int64_t> cube({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}});
  sharding_spec->sharding = xla::HloSharding::Tile(cube).ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 1, 2}));

  // Replicated, all shards should be identical.
  sharding_spec->sharding = xla::HloSharding::Replicate().ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 7, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 7, 4}));

  // 4D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array4D<int64_t> tesseract({{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}});
  sharding = xla::HloSharding::Tile(tesseract).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 1, 2}));

  // 4D tiled and padded, all shard sizes should be idential.
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));

  // 5D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({10, 1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array<int64_t> hypercube(std::vector<int64_t>{1, 1, 2, 2, 2});
  hypercube.FillIota(0);
  sharding = xla::HloSharding::Tile(hypercube).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 3, 2}));

  // 5D tiled and padded, all shard sizes should be identical.
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
}

TEST_F(XLAShardingTest, ShardTensorLocalMesh) {
  // Test sharding with a local mesh.
  std::vector<std::string> devices = {"TPU:8",  "TPU:9",  "TPU:10", "TPU:11",
                                      "TPU:12", "TPU:13", "TPU:14", "TPU:15"};

  // 1D tiled
  at::Tensor tensor = at::ones({8}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::OpSharding sharding =
      xla::HloSharding::Tile1D(
          CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice()),
          devices.size())
          .ToProto();
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  for (auto shard : shards) {
    EXPECT_EQ(shard.sizes(), c10::ArrayRef<long>({1}));
  }

  // 2D tiled, The first dim is halved and the last replicated. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {0, 1, 2, 3},
      {4, 5, 6, 7},
  });
  sharding = xla::HloSharding::Tile(mesh).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({4, 1, 4}));

  // 3D tiled, the first dim is replicated and the last halved. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  xla::Array3D<int64_t> cube({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}});
  sharding_spec->sharding = xla::HloSharding::Tile(cube).ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 1, 2}));

  // Replicated, all shards should be identical.
  sharding_spec->sharding = xla::HloSharding::Replicate().ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 7, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 7, 4}));

  // 4D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array4D<int64_t> tesseract({{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}});
  sharding = xla::HloSharding::Tile(tesseract).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 1, 2}));

  // 4D tiled and padded, all shard sizes should be idential.
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));

  // 5D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({10, 1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array<int64_t> hypercube(std::vector<int64_t>{1, 1, 2, 2, 2});
  hypercube.FillIota(0);
  sharding = xla::HloSharding::Tile(hypercube).ToProto();
  sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 3, 2}));

  // 5D tiled and padded, all shard sizes should be identical.
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                     /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
}

TEST_F(XLAShardingTest, ShardTensorMultiHost) {
  std::vector<std::string> devices = {"TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // 2D tiled, The first dim is halved and the last replicated.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  xla::Array2D<int64_t> mesh({
      {4, 5, 0, 1},
      {6, 7, 2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
  auto sharding_spec =
      std::make_shared<XLATensor::ShardingSpec>(sharding, tensor_shape);
  // For devices at the start of the mesh, all shards should have the same
  // unpadded shape.
  auto shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
                                          /*padded=*/false);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({4, 2, 4}));

  // When this host's devices are at the end of the mesh, the last shard should
  // be smaller in dim=2 because it's not evenly divisible.
  mesh = xla::Array2D<int64_t>({
      {0, 1, 4, 5},
      {2, 3, 6, 7},
  });
  sharding_spec->sharding = xla::HloSharding::Tile(mesh).ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding_spec, devices,
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

  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
  auto sharding_spec = std::make_shared<XLATensor::ShardingSpec>(
      sharding, global_shape, /*minibatch=*/true);
  auto shards = ShardingUtil::ShardTensor(minibatch_tensor, sharding_spec,
                                          devices, /*padded=*/true);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({2, 7, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({2, 7, 4}));
}

TEST_F(XLAShardingTest, EqualShardingSpecs) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, bridge::GetDefaultDevice());
  XLATensor::ShardingSpec tiled_2d(xla::HloSharding::Tile({
                                                              {0, 1, 2, 3},
                                                              {4, 5, 6, 7},
                                                          })
                                       .ToProto(),
                                   tensor_shape);
  XLATensor::ShardingSpec tiled_3d(
      xla::HloSharding::Tile({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}).ToProto(),
      tensor_shape);
  XLATensor::ShardingSpec replicated(xla::HloSharding::Replicate().ToProto(),
                                     tensor_shape);
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_2d));
  EXPECT_FALSE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_3d));
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(replicated, replicated));
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
  std::vector<XLATensor::ShardingSpecPtr> shardings = {
      nullptr,
      std::make_shared<XLATensor::ShardingSpec>(
          xla::HloSharding::Replicate().ToProto(), tensor_shape),
      std::make_shared<XLATensor::ShardingSpec>(
          xla::HloSharding::Unknown().ToProto(), tensor_shape)};
  std::vector<torch::lazy::BackendDataPtr> tensors_data =
      CreateTensorsData(tensors, shardings, devices);

  int64_t n_devices =
      torch_xla::runtime::GetComputationClient()->GetLocalDevices().size();
  if (n_devices > 1) {
    // null sharding is treated as replicated.
    auto xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[0]);
    std::vector<torch_xla::runtime::ComputationClient::DataPtr> shards =
        torch_xla::runtime::GetComputationClient()->GetDataShards(xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(tensors_data[0], shards[0], at::kFloat));

    // Returns multiple input shards, explicitly replicated
    auto sharded_xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[1]);
    shards = torch_xla::runtime::GetComputationClient()->GetDataShards(
        sharded_xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(sharded_xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(shards[0], shards[1], at::kFloat));

    // Returns multiple input shards, implicitly replicated
    sharded_xla_data =
        std::dynamic_pointer_cast<torch_xla::runtime::ComputationClient::Data>(
            tensors_data[2]);
    shards = torch_xla::runtime::GetComputationClient()->GetDataShards(
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
      torch_xla::runtime::GetComputationClient()->GetLocalDevices().size();
  xla::Array<int64_t> tile_assignment({1, n_devices});
  tile_assignment.FillIota(0);
  xla::OpSharding tiled = xla::HloSharding::Tile(tile_assignment).ToProto();

  // Build simple addition with a sharded input.
  xla::XlaBuilder b("builder");
  b.SetSharding(tiled);
  auto x = xla::Parameter(&b, 0, shape, "p0");
  b.ClearSharding();
  auto y = xla::Add(x, xla::ConstantR0<float>(&b, 3));
  xla::XlaComputation xla_computation =
      ConsumeValue(b.Build(/*remove_dynamic_dimensions=*/false));
  std::vector<torch_xla::runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(xla_computation),
                       bridge::GetDefaultDevice()->toString(),
                       {bridge::GetDefaultDevice()->toString()},
                       &shape,
                       /*should_wrap_parameter=*/false,
                       /*is_sharded=*/true});

  std::vector<
      std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>>
      computations = torch_xla::runtime::GetComputationClient()->Compile(
          std::move(instances));
  torch_xla::runtime::ComputationClient::ComputationPtr computation =
      std::make_shared<torch_xla::runtime::ComputationClient::Computation>(
          "add", std::move(computations[0]->move_computation()));

  // Prepare output sharding propagation, expect a sharded output placeholder.
  std::vector<XLATensorPtr> tensors{XLATensor::Create(
      torch_xla::runtime::GetComputationClient()->CreateDataPlaceholder(
          bridge::GetDefaultDevice()->toString(), std::move(shape)))};
  std::vector<torch::lazy::BackendDataPtr> data_placeholders;
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs;
  ShardingUtil::PrepareOutputShardingPropagation(
      &tensors, {0}, computation, &data_placeholders, &sharding_specs);

  // Check if the output sharding spec is correctly extracted.
  EXPECT_EQ(sharding_specs.size(), 1);
  if (n_devices > 1) {
    // Tiled sharding requires multiple devices.
    EXPECT_TRUE(
        xla::protobuf_util::ProtobufEquals(tiled, sharding_specs[0]->sharding));
  } else {
    // Sincle device execution defaults to replication sharding.
    EXPECT_TRUE(xla::protobuf_util::ProtobufEquals(
        xla::HloSharding::Replicate().ToProto(), sharding_specs[0]->sharding));
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
