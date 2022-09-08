#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <iostream>

#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_sharding_util.h"
#include "torch_xla_test.h"

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
  auto shards = ShardingUtil::ShardTensor(tensor, sharding, devices);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({1}));
  EXPECT_EQ(shards[1].sizes(), c10::ArrayRef<long>({1}));

  // 2D tiled, the last shard size should be smaller in dim=1
  tensor = at::ones({8, 7, 4}, at::TensorOptions(at::kFloat));
  xla::Array2D<int64_t> mesh({
      {0, 1, 2, 3},
      {4, 5, 6, 7},
  });
  sharding = xla::HloSharding::Tile(mesh).ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding, devices);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({4, 2, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({4, 1, 4}));

  // Replicated, all shards should be identical
  sharding = xla::HloSharding::Replicate().ToProto();
  shards = ShardingUtil::ShardTensor(tensor, sharding, devices);
  EXPECT_EQ(shards.size(), 8);
  EXPECT_EQ(shards[0].sizes(), c10::ArrayRef<long>({8, 7, 4}));
  EXPECT_EQ(shards[7].sizes(), c10::ArrayRef<long>({8, 7, 4}));
}

}  // namespace cpp_test
}  // namespace torch_xla
