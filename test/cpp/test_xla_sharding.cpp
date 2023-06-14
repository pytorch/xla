#include <ATen/ATen.h>
#include <google/protobuf/repeated_field.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <iostream>

#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_data.h"
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

TEST_F(XLAShardingTest, GetShardShape) {
  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
  auto shard_shape = ShardingUtil::GetShardShape(tensor, sharding);
  // For tiled sharding, each dimension should be halved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({4, 4}));

  sharding = xla::HloSharding::Replicate().ToProto();
  shard_shape = ShardingUtil::GetShardShape(tensor, sharding);
  // For replicated sharding, each dimension should be preserved
  EXPECT_EQ(shard_shape, std::vector<int64_t>({8, 7}));
}

TEST_F(XLAShardingTest, GetShardIndicesForDevices) {
  std::vector<std::string> devices = {"TPU:0", "TPU:1", "TPU:2", "TPU:3"};

  auto tensor = at::ones({8, 7}, at::TensorOptions(at::kFloat));
  xla::Array2D<int64_t> mesh({
      {0, 1},
      {2, 3},
  });
  auto sharding = xla::HloSharding::Tile(mesh).ToProto();
  auto shard_shape = ShardingUtil::GetShardShape(tensor, sharding);
  auto shard_indices = ShardingUtil::GetShardIndicesForDevices(
      shard_shape, tensor.sizes().vec(), sharding, devices);
  EXPECT_EQ(shard_indices.size(), devices.size());
  /* Tiled indices should be:
                 dim=0 dim=1
       device=0  [0:4,  0:4]
       device=1  [0:4,  4:7]
       device=2  [4:8,  0:4]
       device=3  [4:8,  4:7] */
  std::vector<std::vector<int>> slice_starts = {{0, 0}, {0, 4}, {4, 0}, {4, 4}};
  std::vector<std::vector<int>> slice_ends = {{4, 4}, {4, 7}, {8, 4}, {8, 7}};
  for (int device = 0; device < shard_indices.size(); ++device) {
    EXPECT_EQ(shard_indices[device].size(), tensor.sizes().size());
    for (int dim = 0; dim < shard_indices[device].size(); ++dim) {
      EXPECT_TRUE(shard_indices[device][dim].is_slice());
      auto slice = shard_indices[device][dim].slice();
      EXPECT_EQ(slice.start(), slice_starts[device][dim]);
      EXPECT_EQ(slice.stop(), slice_ends[device][dim]);
      EXPECT_EQ(slice.step(), 1);
    }
  }

  sharding = xla::HloSharding::Replicate().ToProto();
  shard_shape = ShardingUtil::GetShardShape(tensor, sharding);
  shard_indices = ShardingUtil::GetShardIndicesForDevices(
      shard_shape, tensor.sizes().vec(), sharding, devices);
  EXPECT_EQ(shard_indices.size(), devices.size());
  for (int i = 0; i < devices.size(); ++i) {
    EXPECT_EQ(shard_indices[i].size(), 1);
    EXPECT_TRUE(shard_indices[i][0].is_ellipsis());
  }
}

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
  if (torch_xla::runtime::sys_util::GetEnvString(
          torch_xla::runtime::env::kEnvPjRtDevice, "") == "") {
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
  std::vector<torch_xla::runtime::ComputationClient::DataPtr> shards =
      torch_xla::runtime::GetComputationClient()->GetDataShards(xla_data);
  EXPECT_EQ(shards.size(), 1);
  EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(xla_data->shape(),
                                                 shards[0]->shape()));
  EXPECT_TRUE(
      XlaDataValuesEqual(tensors_data[0], WrapXlaData(shards[0]), at::kFloat));

  // Returns multiple input shards, replicated
  int64_t n_devices =
      torch_xla::runtime::GetComputationClient()->GetLocalDevices().size();
  if (n_devices > 1) {
    auto sharded_xla_data =
        dynamic_cast<XLAData*>(tensors_data[1].get())->xla_data();
    shards = torch_xla::runtime::GetComputationClient()->GetDataShards(
        sharded_xla_data);
    EXPECT_EQ(shards.size(), n_devices);
    EXPECT_TRUE(xla::Shape::Equal().IgnoreLayout()(sharded_xla_data->shape(),
                                                   shards[0]->shape()));
    EXPECT_TRUE(XlaDataValuesEqual(WrapXlaData(shards[0]),
                                   WrapXlaData(shards[1]), at::kFloat));
  }
}

TEST_F(XLAShardingTest, InputHandler) {
  if ((torch_xla::runtime::sys_util::GetEnvString(
           torch_xla::runtime::env::kEnvPjRtDevice, "") == "") ||
      (torch_xla::runtime::GetComputationClient()->GetLocalDevices().size() <
       2)) {
    GTEST_SKIP()
        << "`PJRT_DEVICE` is not set, with more than 2 local devices, ("
        << torch_xla::runtime::GetComputationClient()->GetLocalDevices().size()
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

  std::vector<torch_xla::runtime::ComputationClient::DataPtr> arguments =
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
  if ((torch_xla::runtime::sys_util::GetEnvString(
           torch_xla::runtime::env::kEnvPjRtDevice, "") == "") ||
      (torch_xla::runtime::GetComputationClient()->GetLocalDevices().size() <
       2)) {
    GTEST_SKIP()
        << "`PJRT_DEVICE` is not set, with more than 2 local devices, ("
        << torch_xla::runtime::GetComputationClient()->GetLocalDevices().size()
        << " local devices detected).";
  }

  std::vector<std::string> devices =
      torch_xla::runtime::GetComputationClient()->GetLocalDevices();

  // Prepare an input vecotr `outputs` with 2 arguments per device.
  std::vector<std::vector<torch_xla::runtime::ComputationClient::DataPtr>>
      outputs;
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
  std::vector<torch_xla::runtime::ComputationClient::DataPtr> sharded_outputs =
      ShardingUtil::OutputHandler(outputs, sharding_specs,
                                  /*replicated_output=*/true);
  EXPECT_EQ(sharded_outputs.size(), 2);
  auto shards = torch_xla::runtime::GetComputationClient()->GetDataShards(
      sharded_outputs[0]);
  EXPECT_EQ(shards.size(), devices.size());
  EXPECT_FALSE(
      xla::Shape::Equal().IgnoreLayout()(shards[0]->shape(), tensor_shape));

  // Wrap sharded data into a PjRtShardedData with `devices.size()` shards.
  std::vector<torch_xla::runtime::ComputationClient::DataPtr> wrapped_outputs =
      ShardingUtil::OutputHandler(outputs, sharding_specs,
                                  /*replicated_output=*/false);
  EXPECT_EQ(wrapped_outputs.size(), 2);
  shards = torch_xla::runtime::GetComputationClient()->GetDataShards(
      wrapped_outputs[0]);
  EXPECT_EQ(shards.size(), devices.size());
  EXPECT_TRUE(
      xla::Shape::Equal().IgnoreLayout()(shards[0]->shape(), tensor_shape));
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
                       GetDefaultDevice()->toString(),
                       {GetDefaultDevice()->toString()},
                       &shape,
                       /*should_wrap_parameter=*/false,
                       /*is_sharded=*/true});

  std::vector<
      std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>>
      computations = torch_xla::runtime::GetComputationClient()->Compile(
          std::move(instances));
  ComputationPtr computation = std::make_shared<Computation>(
      "add", std::move(computations[0]->move_computation()));

  // Prepare output sharding propagation, expect a sharded output placeholder.
  std::vector<XLATensorPtr> tensors{XLATensor::Create(WrapXlaData(
      torch_xla::runtime::GetComputationClient()->CreateDataPlaceholder(
          GetDefaultDevice()->toString(), std::move(shape))))};
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
  EXPECT_EQ(UnwrapXlaData(data_placeholders[0])->device(), "SPMD:0");
  EXPECT_FALSE(UnwrapXlaData(data_placeholders[0])->HasValue());
}

}  // namespace cpp_test
}  // namespace torch_xla
