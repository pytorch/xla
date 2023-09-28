#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <iostream>

#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/multi_wait.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/thread_pool.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "xla/client/xla_builder.h"
#include "xla/shape_util.h"

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
  std::vector<torch_xla::runtime::ComputationClient::CompileInstance> instances;
  for (auto& device_str : device_strings) {
    instances.emplace_back(CreateCrsComputation(shape), device_str,
                           all_device_strings, &shape);
  }
  auto compiled_computations =
      torch_xla::runtime::GetComputationClient()->Compile(std::move(instances));

  std::vector<at::Tensor> tensors;
  for (size_t i = 0; i < device_strings.size(); ++i) {
    tensors.push_back(at::ones({8, 8}, at::TensorOptions(at::kFloat)));
  }
  auto tensors_data = CreateTensorsData(tensors, device_strings);

  std::vector<std::vector<torch_xla::runtime::ComputationClient::DataPtr>>
      results(device_strings.size());
  torch_xla::runtime::util::MultiWait mwait(device_strings.size());
  torch_xla::runtime::ComputationClient::ExecuteComputationOptions exec_options;
  for (size_t i = 0; i < device_strings.size(); ++i) {
    auto executor = [&, i]() {
      results[i] =
          torch_xla::runtime::GetComputationClient()->ExecuteComputation(
              *compiled_computations[i],
              {std::dynamic_pointer_cast<
                  torch_xla::runtime::ComputationClient::Data>(
                  tensors_data[i])},
              device_strings[i], exec_options);
    };
    torch_xla::runtime::env::ScheduleIoClosure(
        mwait.Completer(std::move(executor)));
  }
  mwait.Wait();

  for (size_t i = 0; i < results.size(); ++i) {
    auto literals =
        torch_xla::runtime::GetComputationClient()->TransferFromServer(
            results[i]);
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
