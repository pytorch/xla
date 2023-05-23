#ifndef XLA_TEST_CPP_CPP_TEST_UTIL_H_
#define XLA_TEST_CPP_CPP_TEST_UTIL_H_

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <functional>
#include <string>
#include <unordered_set>

#include "absl/types/span.h"
#include "third_party/xla_client/computation_client.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/tensor.h"

#define XLA_CPP_TEST_ENABLED(name)                          \
  do {                                                      \
    if (!::torch_xla::DebugUtil::ExperimentEnabled(name)) { \
      GTEST_SKIP();                                         \
    }                                                       \
  } while (0)

namespace torch_xla {
namespace cpp_test {

const std::unordered_set<std::string>* GetIgnoredCounters();

// Converts an at::Tensor(device=torch::kXLA) to at::Tensor(device=torch::kCPU)
// This at::Tensor can be torch::Tensor which is a Variable, or at::Tensor which
// know nothing about autograd. If the input tensor is already a CPU tensor, it
// will be returned. Needed because EqualValues and AllClose require CPU tensors
// on both sides.
at::Tensor ToCpuTensor(const at::Tensor& tensor);

// Helper function to copy a tensor to device.
torch::Tensor CopyToDevice(const torch::Tensor& tensor,
                           const torch::Device& device,
                           bool requires_grad = false);

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2);

bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2);

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol = 1e-5,
                 double atol = 1e-8);

static inline void AllClose(at::Tensor tensor, at::Tensor xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, xla_tensor, rtol, atol));
}

static inline void AllClose(at::Tensor tensor, XLATensorPtr& xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, xla_tensor->ToTensor(/*detached=*/false),
                          rtol, atol));
}

static inline void AllEqual(at::Tensor tensor, at::Tensor xla_tensor) {
  EXPECT_TRUE(EqualValues(tensor, xla_tensor));
}

void ForEachDevice(
    absl::Span<const DeviceType> device_types,
    const std::function<void(const torch::lazy::BackendDevice&)>& devfn);

void ForEachDevice(absl::Span<const DeviceType> device_types,
                   const std::function<void(const torch::Device&)>& devfn);

void ForEachDevice(
    const std::function<void(const torch::lazy::BackendDevice&)>& devfn);

void ForEachDevice(const std::function<void(const torch::Device&)>& devfn);

void WithAllDevices(
    absl::Span<const DeviceType> device_types,
    const std::function<void(const std::vector<torch::lazy::BackendDevice>&,
                             const std::vector<torch::lazy::BackendDevice>&)>&
        devfn);

std::string GetTensorTextGraph(at::Tensor tensor);

std::string GetTensorDotGraph(at::Tensor tensor);

std::string GetTensorHloGraph(at::Tensor tensor);

torch::lazy::Value GetTensorIrValue(const at::Tensor& tensor,
                                    const torch::lazy::BackendDevice& device);

std::vector<xla::ComputationClient::DataPtr> Execute(
    absl::Span<const torch::lazy::Value> roots,
    const torch::lazy::BackendDevice& device);

std::vector<at::Tensor> Fetch(
    absl::Span<const xla::ComputationClient::DataPtr> device_data);

std::vector<at::Tensor> ExecuteAndFetch(
    absl::Span<const torch::lazy::Value> roots,
    const torch::lazy::BackendDevice& device);

void AssertBackward(const torch::Tensor& xla_output,
                    const std::vector<torch::Tensor>& xla_inputs,
                    const torch::Tensor& reference_output,
                    const std::vector<torch::Tensor>& reference_inputs);

void TestBackward(
    const std::vector<torch::Tensor>& inputs, const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol = 1e-5, double atol = 1e-8, int derivative_level = 1);

torch::lazy::NodePtr CreateNonZeroNode2d(int64_t num_non_zero_element,
                                         int64_t num_row, int64_t num_col);

bool UsingPjRt();

bool UsingTpu();

}  // namespace cpp_test
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_CPP_TEST_UTIL_H_