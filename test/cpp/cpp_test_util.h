#pragma once

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace cpp_test {

// Converts an XLA ATen tensor to a CPU backend tensor. Extracts it first from
// an autograd variable, if needed. Needed because EqualValues and AllClose
// require CPU tensors on both sides. If the input tensor is already a CPU
// tensor, it will be returned.
at::Tensor ToCpuTensor(const at::Tensor& t);

at::Tensor ToTensor(XLATensor& xla_tensor);

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2);

bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2);

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol = 1e-5,
                 double atol = 1e-8);

static inline void AllClose(at::Tensor tensor, at::Tensor xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, xla_tensor, rtol, atol));
}

static inline void AllClose(at::Tensor tensor, XLATensor& xla_tensor,
                            double rtol = 1e-5, double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, ToTensor(xla_tensor), rtol, atol));
}

void ForEachDevice(const std::function<void(const Device&)>& devfn);

void WithAllDevices(
    DeviceType device_type,
    const std::function<void(const std::vector<Device>&,
                             const std::vector<Device>&)>& devfn);

std::string GetTensorTextGraph(at::Tensor tensor);

std::string GetTensorDotGraph(at::Tensor tensor);

ir::Value GetTensorIrValue(const at::Tensor& tensor, const Device& device);

std::vector<xla::ComputationClient::DataPtr> Execute(
    tensorflow::gtl::ArraySlice<const ir::Value> roots, const Device& device);

std::vector<at::Tensor> Fetch(
    tensorflow::gtl::ArraySlice<const xla::ComputationClient::DataPtr>
        device_data);

std::vector<at::Tensor> ExecuteAndFetch(
    tensorflow::gtl::ArraySlice<const ir::Value> roots, const Device& device);

}  // namespace cpp_test
}  // namespace torch_xla
