#include "torch_xla/csrc/runtime/pjrt_computation_client.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/tensor_source.h"
#include "torch_xla/csrc/status.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"

namespace torch_xla {
namespace runtime {

class PjRtComputationClientTest : public ::testing::Test {
 protected:
  PjRtComputationClientTest() {
    // Get a CPU client.
    tsl::setenv("PJRT_DEVICE", "CPU", true);
    client_ = GetValueOrThrow(PjRtComputationClient::Create());
    device_ = client_->GetDefaultDevice();
  }

  static void FakeXlaCompileForTesting(
      PjRtComputationClient* client,
      std::function<absl::Status()> fake_compile) {
    client->FakeXlaCompileForTesting(std::move(fake_compile));
  }

  std::unique_ptr<PjRtComputationClient> client_;
  std::string device_;
};

// Returns a computation to compute x + y where x and y are both F32[2,2]
// arrays.
absl::StatusOr<xla::XlaComputation> MakeAddComputation() {
  const xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  return builder.Build();
}

TEST_F(PjRtComputationClientTest, ThrowsExpectedExceptionWhenCompileFails) {
  // Compose a computation to add two matrices.
  xla::Shape out_shape(xla::F32, {2, 2},
                       /*dynamic_dimensions=*/{});
  std::vector<ComputationClient::CompileInstance> instances;
  instances.push_back(ComputationClient::CompileInstance(
      std::move(MakeAddComputation().value()), device_,
      client_->GetCompilationDevices(device_, client_->GetLocalDevices()),
      &out_shape));

  // Force XLA to fail with the given error when invoked by Compile() below.
  FakeXlaCompileForTesting(
      client_.get(), [] { return absl::InvalidArgumentError("invalid arg"); });

  // Compiling the graph should fail, which should throw instead of crashing.
  EXPECT_THROW(client_->Compile(std::move(instances)), std::invalid_argument);
}

TEST_F(PjRtComputationClientTest, ThrowsExpectedExceptionWhenCompileThrows) {
  // Compose a computation to add two matrices.
  xla::Shape out_shape(xla::F32, {2, 2},
                       /*dynamic_dimensions=*/{});
  std::vector<ComputationClient::CompileInstance> instances;
  instances.push_back(ComputationClient::CompileInstance(
      std::move(MakeAddComputation().value()), device_,
      client_->GetCompilationDevices(device_, client_->GetLocalDevices()),
      &out_shape));

  // Force XLA to throw with the given error when invoked by Compile() below.
  FakeXlaCompileForTesting(client_.get(), []() -> absl::Status {
    throw absl::BadStatusOrAccess(absl::InvalidArgumentError("invalid arg"));
  });

  // Compiling the graph should fail, which should throw instead of crashing.
  EXPECT_THROW(client_->Compile(std::move(instances)), std::invalid_argument);
}

TEST_F(PjRtComputationClientTest, Init) {
  // Compose a computation to add two 2x2 matrices.
  auto out_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  std::vector<ComputationClient::CompileInstance> instances;
  instances.push_back(ComputationClient::CompileInstance(
      std::move(MakeAddComputation().value()), device_,
      client_->GetCompilationDevices(device_, client_->GetLocalDevices()),
      &out_shape));

  // Prepare inputs.
  xla::Literal literal_x =
      xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  xla::Literal literal_y =
      xla::LiteralUtil::CreateR2<float>({{5.0f, 6.0f}, {7.0f, 8.0f}});

  // Compile the graph.
  std::vector<ComputationClient::ComputationPtr> computations =
      client_->Compile(std::move(instances));

  // Copy inputs to device.
  ComputationClient::ExecuteComputationOptions options{};
  std::vector<std::shared_ptr<const TensorSource>> args = {
      std::make_shared<LiteralSource>(std::move(literal_x), device_),
      std::make_shared<LiteralSource>(std::move(literal_y), device_)};

  // Execute the graph.
  std::vector<ComputationClient::DataPtr> results =
      GetValueOrThrow(client_->ExecuteComputation(
          *computations[0],
          client_->TransferToDevice(absl::MakeConstSpan(args)), device_,
          options));

  // Copy the output from device back to host and assert correctness.
  ASSERT_EQ(results.size(), 1);
  auto result_literals = GetValueOrThrow(client_->TransferFromDevice(results));
  ASSERT_THAT(result_literals, ::testing::SizeIs(1));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR2<float>({{6.0f, 8.0f}, {10.0f, 12.0f}}),
      result_literals[0]));
}

}  // namespace runtime
}  // namespace torch_xla
