#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <iostream>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/multi_wait.h"
#include "third_party/xla_client/thread_pool.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace cpp_test {
namespace {

xla::XlaComputation CreateCrsComputation(const xla::Shape& shape) {
  xla::XlaBuilder builder("CrsComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, shape, "x");
  xla::CrossReplicaSum(x);
  return ConsumeValue(builder.Build());
}

void TestSingleReplication(
    const std::vector<torch::lazy::BackendDevice>& devices,
    const std::vector<torch::lazy::BackendDevice>& all_devices) {
  // Simulates N threads executing the same computation, using separated XRT
  // executions, and issuing CRS operations.
  std::vector<std::string> device_strings;
  std::vector<std::string> all_device_strings;
  for (auto& device : devices) {
    device_strings.push_back(device.toString());
  }
  for (auto& device : all_devices) {
    all_device_strings.push_back(device.toString());
  }
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {8, 8});
  std::vector<xla::ComputationClient::CompileInstance> instances;
  for (auto& device_str : device_strings) {
    instances.emplace_back(CreateCrsComputation(shape), device_str,
                           all_device_strings, &shape);
  }
  auto compiled_computations =
      xla::ComputationClient::Get()->Compile(std::move(instances));

  std::vector<at::Tensor> tensors;
  for (size_t i = 0; i < device_strings.size(); ++i) {
    tensors.push_back(at::ones({8, 8}, at::TensorOptions(at::kFloat)));
  }
  auto tensors_data = CreateTensorsData(tensors, device_strings);

  std::vector<std::vector<xla::ComputationClient::DataPtr>> results(
      device_strings.size());
  xla::util::MultiWait mwait(device_strings.size());
  xla::ComputationClient::ExecuteComputationOptions exec_options;
  for (size_t i = 0; i < device_strings.size(); ++i) {
    auto executor = [&, i]() {
      results[i] = xla::ComputationClient::Get()->ExecuteComputation(
          *compiled_computations[i], {UnwrapXlaData(tensors_data[i])},
          device_strings[i], exec_options);
    };
    xla::env::ScheduleIoClosure(mwait.Completer(std::move(executor)));
  }
  mwait.Wait();

  for (size_t i = 0; i < results.size(); ++i) {
    auto literals =
        xla::ComputationClient::Get()->TransferFromServer(results[i]);
    ASSERT_EQ(literals.size(), 1);

    // The result must be the original tensor value, multiplied by the number of
    // devices into which we replicated.
    at::Tensor result =
        MakeTensorFromXlaLiteral(literals.front(), tensors[i].scalar_type());
    AllClose(result, tensors[i] * static_cast<float>(all_devices.size()));
  }
}

}  // namespace

class ReplicationTest : public AtenXlaTensorTestBase {};

TEST_F(ReplicationTest, TestNSingleReplication) {
  WithAllDevices(
      {XlaDeviceType::TPU, XlaDeviceType::GPU},
      [&](const std::vector<torch::lazy::BackendDevice>& devices,
          const std::vector<torch::lazy::BackendDevice>& all_devices) {
        TestSingleReplication(devices, all_devices);
      });
}

}  // namespace cpp_test
}  // namespace torch_xla
