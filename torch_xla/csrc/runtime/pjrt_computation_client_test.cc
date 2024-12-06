#include "torch_xla/csrc/runtime/pjrt_computation_client.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/runtime/tensor_source.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace torch_xla {
namespace runtime {

absl::StatusOr<xla::XlaComputation> MakeComputation() {
  xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  return builder.Build();
}

TEST(PjRtComputationClientTest, Init) {
  // Get a CPU client.
  tsl::setenv("PJRT_DEVICE", "CPU", true);
  auto client = std::make_unique<PjRtComputationClient>();
  std::string device = client->GetDefaultDevice();

  // Compose a computation.
  auto shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  std::vector<ComputationClient::CompileInstance> instances;
  instances.push_back(ComputationClient::CompileInstance(
      std::move(MakeComputation().value()), device,
      client->GetCompilationDevices(device, client->GetLocalDevices()),
      &shape));

  // Prepare inputs.
  xla::Literal literal_x =
      xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  xla::Literal literal_y =
      xla::LiteralUtil::CreateR2<float>({{5.0f, 6.0f}, {7.0f, 8.0f}});

  // Compile the graph.
  std::vector<ComputationClient::ComputationPtr> computations =
      client->Compile(std::move(instances));

  // Copy inputs to device.
  ComputationClient::ExecuteComputationOptions options{};
  std::vector<std::shared_ptr<const TensorSource>> args = {
      std::make_shared<LiteralSource>(std::move(literal_x), device),
      std::make_shared<LiteralSource>(std::move(literal_y), device)};

  // Execute the graph.
  std::vector<ComputationClient::DataPtr> results = client->ExecuteComputation(
      *computations[0], client->TransferToDevice(absl::MakeConstSpan(args)),
      device, options);

  // Copy the output from device back to host and assert correctness..
  ASSERT_EQ(results.size(), 1);
  auto result_literals = client->TransferFromDevice(results);
  ASSERT_THAT(result_literals, ::testing::SizeIs(1));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR2<float>({{6.0f, 8.0f}, {10.0f, 12.0f}}),
      result_literals[0]));
}

}  // namespace runtime
}  // namespace torch_xla
