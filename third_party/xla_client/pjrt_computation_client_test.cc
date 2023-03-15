#include "third_party/xla_client/pjrt_computation_client.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/third_party/tsl/platform/logging.h"
#include "xla/third_party/tsl/lib/core/status_test_util.h"
#include "xla/third_party/tsl/platform/env.h"
#include "xla/third_party/tsl/platform/test.h"
#include "third_party/xla_client/computation_client.h"

namespace xla {

tsl::StatusOr<xla::XlaComputation> MakeComputation() {
  xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  return builder.Build();
}

ComputationClient::TensorSource TensorSourceFromLiteral(
    const std::string& device, const xla::Literal& literal) {
  auto populate_fn = [&](const ComputationClient::TensorSource& source_tensor,
                         void* dest_buffer, size_t dest_buffer_size) {
    std::memcpy(dest_buffer, literal.data<float>().data(),
                dest_buffer_size * sizeof(literal.data<float>().data()));
  };
  return ComputationClient::TensorSource(literal.shape(), device,
                                         std::move(populate_fn));
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
  std::vector<ComputationClient::TensorSource> args = {
      TensorSourceFromLiteral(device, literal_x),
      TensorSourceFromLiteral(device, literal_y)};

  // Execute the graph.
  std::vector<ComputationClient::DataPtr> results = client->ExecuteComputation(
      *computations[0], client->TransferToServer(absl::MakeConstSpan(args)),
      device, options);

  // Copy the output from device back to host and assert correctness..
  ASSERT_EQ(results.size(), 1);
  auto result_literals = client->TransferFromServer(results);
  ASSERT_THAT(result_literals, ::testing::SizeIs(1));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR2<float>({{6.0f, 8.0f}, {10.0f, 12.0f}}),
      result_literals[0]));
}

}  // namespace xla
