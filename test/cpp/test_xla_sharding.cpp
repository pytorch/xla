#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <iostream>

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/sys_util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {
namespace cpp_test {
namespace {
bool XlaDataValuesEqual(torch::lazy::BackendDataPtr a,
                        torch::lazy::BackendDataPtr b,
                        at::ScalarType element_type) {
  std::vector<at::Tensor> tensors = XlaDataToTensors({a, b}, element_type);
  return TensorCompare(tensors[0], tensors[1]);
}
}  // namespace

class XLAShardingTest : public AtenXlaTensorTestBase {};

TEST_F(XLAShardingTest, ShardTensor) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3",
                                      "TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // 1D tiled
  at::Tensor tensor = at::ones({8}, at::TensorOptions(at::kFloat));
  xla::OpSharding sharding =
      xla::HloSharding::Tile1D(
          CreateComputationShapeFromTensor(tensor, GetDefaultDevice()),
          devices.size())
          .ToProto();
  auto shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1}));
  EXPECT_EQ(shards[1].sizes(), c10::ArrayRef<long>({1}));

  // 2D tiled, The first dim is halved and the last replicated. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Array2D<int64_t> mesh({
      {0, 1, 2, 3},
      {4, 5, 6, 7},
  });
  sharding = xla::HloSharding::Tile(mesh).ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({4, 1, 4}));

  // 3D tiled, the first dim is replicated and the last halved. The last shard
  // size should be smaller in dim=1 because it's not evenly divisible.
  xla::Array3D<int64_t> cube({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}});
  sharding = xla::HloSharding::Tile(cube).ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 1, 2}));

  // Replicated, all shards should be identical.
  sharding = xla::HloSharding::Replicate().ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 7, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 7, 4}));

  // 4D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Array4D<int64_t> tesseract({{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}});
  sharding = xla::HloSharding::Tile(tesseract).ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 1, 2}));

  // 4D tiled and padded, all shard sizes should be idential.
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({1, 8, 2, 2}));

  // 5D tiled, the first and second dims are replicated and the last halved. The
  // last shard size should be smaller in dim=2 because it's not evenly
  // divisible.
  tensor = at::ones({10, 1, 8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Array<int64_t> hypercube(std::vector<int64_t>{1, 1, 2, 2, 2});
  hypercube.FillIota(0);
  sharding = xla::HloSharding::Tile(hypercube).ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 3, 2}));

  // 5D tiled and padded, all shard sizes should be identical.
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/true);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({10, 1, 4, 4, 2}));
}

TEST_F(XLAShardingTest, ShardTensorMultiHost) {
  std::vector<std::string> devices = {"TPU:4", "TPU:5", "TPU:6", "TPU:7"};

  // 2D tiled, The first dim is halved and the last replicated.
  at::Tensor tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Array2D<int64_t> mesh({
      {4, 5, 0, 1},
      {6, 7, 2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();

  // For devices at the start of the mesh, all shards should have the same
  // unpadded shape.
  auto shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({4, 2, 4}));

  // When this host's devices are at the end of the mesh, the last shard should
  // be smaller in dim=2 because it's not evenly divisible.
  mesh = xla::Array2D<int64_t>({
      {0, 1, 4, 5},
      {2, 3, 6, 7},
  });
  sharding = xla::HloSharding::Tile(mesh).ToProto();
  shards =
      ShardingUtil::ShardTensor(tensor, sharding, devices, /*padded=*/false);
  EXPECT_EQ(shards.size(), 4);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[3].sizes(), c10::ArrayRef<long>({4, 1, 4}));
}

TEST_F(XLAShardingTest, EqualShardingSpecs) {
  XLATensor::ShardingSpec tiled_2d(xla::HloSharding::Tile({
                                                              {0, 1, 2, 3},
                                                              {4, 5, 6, 7},
                                                          })
                                       .ToProto());
  XLATensor::ShardingSpec tiled_3d(
      xla::HloSharding::Tile({{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}).ToProto());
  XLATensor::ShardingSpec replicated(xla::HloSharding::Replicate().ToProto());
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_2d));
  EXPECT_FALSE(ShardingUtil::EqualShardingSpecs(tiled_2d, tiled_3d));
  EXPECT_TRUE(ShardingUtil::EqualShardingSpecs(replicated, replicated));
  EXPECT_FALSE(ShardingUtil::EqualShardingSpecs(tiled_2d, replicated));
}

TEST_F(XLAShardingTest, CreateTensorsData) {
  if (xla::sys_util::GetEnvString(xla::env::kEnvPjRtDevice, "") == "") {
    GTEST_SKIP() << "`PJRT_DEVICE` is not set.";
  }

  std::vector<at::Tensor> tensors(2);
  std::fill_n(tensors.begin(), tensors.size(),
              at::ones({8, 8}, at::TensorOptions(at::kFloat)));
  std::vector<std::string> devices(2);
  std::fill_n(devices.begin(), devices.size(), GetDefaultDevice()->toString());
  std::vector<XLATensor::ShardingSpecPtr> shardings = {
      nullptr, std::make_shared<XLATensor::ShardingSpec>(
                   xla::HloSharding::Replicate().ToProto())};
  std::vector<torch::lazy::BackendDataPtr> tensors_data =
      CreateTensorsData(tensors, shardings, devices);

  // Returns the input without sharding
  auto xla_data = dynamic_cast<XLAData*>(tensors_data[0].get())->xla_data();
  std::vector<xla::ComputationClient::DataPtr> shards =
      xla::ComputationClient::Get()->GetDataShards(xla_data);
  EXPECT_EQ(shards.size(), 1);
  EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(xla_data->shape(),
                                                 shards[0]->shape()));
  EXPECT_TRUE(
      XlaDataValuesEqual(tensors_data[0], WrapXlaData(shards[0]), at::kFloat));

  // Returns multiple input shards, replicated
  int64_t n_devices = xla::ComputationClient::Get()->GetLocalDevices().size();
  if (n_devices > 1) {
    auto sharded_xla_data =
        dynamic_cast<XLAData*>(tensors_data[1].get())->xla_data();
    shards = xla::ComputationClient::Get()->GetDataShards(sharded_xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(sharded_xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(WrapXlaData(shards[0]),
                                   WrapXlaData(shards[1]), at::kFloat));
  }
}

TEST_F(XLAShardingTest, InputHandler) {
  if ((xla::sys_util::GetEnvString(xla::env::kEnvPjRtDevice, "") == "") ||
      (xla::ComputationClient::Get()->GetLocalDevices().size() < 2)) {
    GTEST_SKIP()
        << "`PJRT_DEVICE` is not set, with more than 2 local devices, ("
        << xla::ComputationClient::Get()->GetLocalDevices().size()
        << " local devices detected).";
  }

  std::vector<at::Tensor> tensors(2);
  std::fill_n(tensors.begin(), tensors.size(),
              at::ones({8, 8}, at::TensorOptions(at::kFloat)));
  std::vector<std::string> devices = {"TPU:0", "TPU:1"};
  std::vector<XLATensor::ShardingSpecPtr> shardings = {
      nullptr, std::make_shared<XLATensor::ShardingSpec>(
                   xla::HloSharding::Replicate().ToProto())};
  std::vector<torch::lazy::BackendDataPtr> tensors_data =
      CreateTensorsData(tensors, shardings, devices);

  std::vector<xla::ComputationClient::DataPtr> arguments =
      UnwrapXlaData(tensors_data);
  auto arguments_by_device = ShardingUtil::InputHandler(arguments, devices);

  auto arg0_dev0 = arguments_by_device[0][0];
  auto arg0_dev1 = arguments_by_device[1][0];
  EXPECT_TRUE(XlaDataValuesEqual(WrapXlaData(arg0_dev0), WrapXlaData(arg0_dev1),
                                 at::kFloat));

  auto arg1_dev0 = arguments_by_device[0][1];
  auto arg1_dev1 = arguments_by_device[1][1];
  EXPECT_TRUE(XlaDataValuesEqual(WrapXlaData(arg1_dev0), WrapXlaData(arg1_dev1),
                                 at::kFloat));
}

TEST_F(XLAShardingTest, OutputHandler) {
  if ((xla::sys_util::GetEnvString(xla::env::kEnvPjRtDevice, "") == "") ||
      (xla::ComputationClient::Get()->GetLocalDevices().size() < 2)) {
    GTEST_SKIP()
        << "`PJRT_DEVICE` is not set, with more than 2 local devices, ("
        << xla::ComputationClient::Get()->GetLocalDevices().size()
        << " local devices detected).";
  }

  std::vector<std::string> devices =
      xla::ComputationClient::Get()->GetLocalDevices();

  // Prepare an input vecotr `outputs` with 2 arguments per device.
  std::vector<std::vector<xla::ComputationClient::DataPtr>> outputs;
  outputs.reserve(devices.size());
  at::Tensor tensor = at::ones({8}, at::TensorOptions(at::kFloat));
  for (auto device : devices) {
    outputs.push_back(
        UnwrapXlaData(CreateTensorsData({tensor, tensor}, {device, device})));
  }

  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, GetDefaultDevice());
  auto sharding_spec = std::make_shared<XLATensor::ShardingSpec>(
      xla::HloSharding::Tile1D(
          CreateComputationShapeFromTensor(tensor, GetDefaultDevice()),
          devices.size())
          .ToProto(),
      tensor_shape);
  std::vector<XLATensor::ShardingSpecPtr> sharding_specs{sharding_spec,
                                                         sharding_spec};

  // Shard a PjRtData into a PjRtShardedData.
  std::vector<xla::ComputationClient::DataPtr> sharded_outputs =
      ShardingUtil::OutputHandler(outputs, sharding_specs,
                                  /*replicated_output=*/true);
  EXPECT_EQ(sharded_outputs.size(), 2);
  auto shards =
      xla::ComputationClient::Get()->GetDataShards(sharded_outputs[0]);
  EXPECT_EQ(shards.size(), devices.size());
  EXPECT_FALSE(
      xla::Shape::Equal().IgnoreLayout()(shards[0]->shape(), tensor_shape));

  // Wrap sharded data into a PjRtShardedData with `devices.size()` shards.
  std::vector<xla::ComputationClient::DataPtr> wrapped_outputs =
      ShardingUtil::OutputHandler(outputs, sharding_specs,
                                  /*replicated_output=*/false);
  EXPECT_EQ(wrapped_outputs.size(), 2);
  shards = xla::ComputationClient::Get()->GetDataShards(wrapped_outputs[0]);
  EXPECT_EQ(shards.size(), devices.size());
  EXPECT_TRUE(
      xla::Shape::Equal().IgnoreLayout()(shards[0]->shape(), tensor_shape));
}

}  // namespace cpp_test
}  // namespace torch_xla
