#include "cpp_test_util.h"

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace cpp_test {
namespace {

void DumpDifferences(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  static bool dump_tensors =
      xla::sys_util::GetEnvBool("XLA_TEST_DUMP_TENSORS", false);
  at::Tensor dtensor1 = tensor1;
  at::Tensor dtensor2 = tensor2;
  if (tensor1.dtype() == at::kBool) {
    dtensor1 = tensor1.toType(at::kByte);
  }
  if (tensor2.dtype() == at::kBool) {
    dtensor2 = tensor2.toType(at::kByte);
  }
  at::Tensor diff = dtensor1 - dtensor2;
  std::cerr << "Difference Tensor:\n" << diff << "\n";
  if (dump_tensors) {
    std::cerr << "Compared Tensors:\n"
              << tensor1 << "\n-vs-\n"
              << tensor2 << "\n";
  }
}

void MaybeDumpGraph(const at::Tensor& tensor) {
  static std::string dump_graph =
      xla::sys_util::GetEnvString("XLA_TEST_DUMP_GRAPHS", "");
  if (!dump_graph.empty() && bridge::IsXlaTensor(tensor)) {
    std::string graph_str;
    if (dump_graph == "text") {
      graph_str = GetTensorTextGraph(tensor);
    } else if (dump_graph == "hlo") {
      graph_str = GetTensorHloGraph(tensor);
    } else if (dump_graph == "dot") {
      graph_str = GetTensorDotGraph(tensor);
    }
    if (!graph_str.empty()) {
      std::cerr << "\n>> Tensor Graph:\n" << graph_str << "\n\n";
    }
  }
}

std::unordered_set<std::string>* CreateIgnoredCounters() {
  std::unordered_set<std::string>* icounters =
      new std::unordered_set<std::string>();
  // Add below the counters whose name need to be ignored when doing
  // is-any-counter-changed assertins.
  icounters->insert("aten::rand");
  return icounters;
}

}  // namespace

const std::unordered_set<std::string>* GetIgnoredCounters() {
  static const std::unordered_set<std::string>* icounters =
      CreateIgnoredCounters();
  return icounters;
}

at::Tensor ToCpuTensor(const at::Tensor& tensor) {
  // tensor.to() implicitly triggers a sync if t.device=torch::kXLA.
  return tensor.to(torch::kCPU);
}

torch::Tensor CopyToDevice(const torch::Tensor& tensor,
                           const torch::Device& device, bool requires_grad) {
  auto output = tensor.to(device, /*non_blocking=*/false, /*copy=*/true);
  if (requires_grad) {
    // The .detach().requires_grad_(true) is to make sure that the new tensor is
    // a leaf in the autograd graph.
    return output.detach().set_requires_grad(true);
  } else {
    return output;
  }
}

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (torch::isnan(tensor1).any().item<bool>()) {
    EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
    tensor1.nan_to_num_();
    tensor2.nan_to_num_();
  }
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }
  bool equal = tensor1.equal(tensor2);
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (tensor1.sizes() != tensor2.sizes()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }
  bool equal = tensor1.equal(tensor2);
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

void ForEachDevice(
    absl::Span<const DeviceType> device_types,
    const std::function<void(const torch::lazy::BackendDevice&)>& devfn) {
  const torch::lazy::BackendDevice* default_device = GetDefaultDevice();
  if (device_types.empty() ||
      std::find_if(device_types.begin(), device_types.end(),
                   [&](const DeviceType device_type) {
                     return device_type.type == default_device->type();
                   }) != device_types.end()) {
    bridge::SetCurrentDevice(*default_device);
    devfn(*default_device);
  } else {
    GTEST_SKIP();
  }
}

void ForEachDevice(absl::Span<const DeviceType> device_types,
                   const std::function<void(const torch::Device&)>& devfn) {
  const torch::lazy::BackendDevice* default_device = GetDefaultDevice();
  if (device_types.empty() ||
      std::find_if(device_types.begin(), device_types.end(),
                   [&](const DeviceType device_type) {
                     return device_type.type == default_device->type();
                   }) != device_types.end()) {
    torch::Device torch_device = bridge::XlaDeviceToAtenDevice(*default_device);
    bridge::SetCurrentDevice(torch_device);
    devfn(torch_device);
  } else {
    GTEST_SKIP();
  }
}

void ForEachDevice(
    const std::function<void(const torch::lazy::BackendDevice&)>& devfn) {
  ForEachDevice({}, devfn);
}

void ForEachDevice(const std::function<void(const torch::Device&)>& devfn) {
  ForEachDevice({}, devfn);
}

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol,
                 double atol) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (torch::isnan(tensor1).any().item<bool>()) {
    EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
    tensor1.nan_to_num_();
    tensor2.nan_to_num_();
  }
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  bool equal = tensor1.allclose(tensor2, rtol, atol);
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

void WithAllDevices(
    absl::Span<const DeviceType> device_types,
    const std::function<void(const std::vector<torch::lazy::BackendDevice>&,
                             const std::vector<torch::lazy::BackendDevice>&)>&
        devfn) {
  for (auto device_type : device_types) {
    std::vector<torch::lazy::BackendDevice> devices;
    std::vector<torch::lazy::BackendDevice> all_devices;
    for (const auto& device_str :
         xla::ComputationClient::Get()->GetLocalDevices()) {
      torch::lazy::BackendDevice device(device_str);
      if (device.type() == device_type.type) {
        devices.push_back(device);
      }
    }
    for (const auto& device_str :
         xla::ComputationClient::Get()->GetAllDevices()) {
      torch::lazy::BackendDevice device(device_str);
      if (device.type() == device_type.type) {
        all_devices.push_back(device);
      }
    }
    if (!devices.empty()) {
      devfn(devices, all_devices);
    }
  }
}

std::string GetTensorTextGraph(at::Tensor tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return ir::DumpUtil::ToText({xtensor.GetIrValue().node.get()});
}

std::string GetTensorDotGraph(at::Tensor tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return ir::DumpUtil::ToDot({xtensor.GetIrValue().node.get()});
}

std::string GetTensorHloGraph(at::Tensor tensor) {
  XLATensor xtensor = bridge::GetXlaTensor(tensor);
  return ir::DumpUtil::ToHlo({xtensor.GetIrValue()}, xtensor.GetDevice());
}

ir::Value GetTensorIrValue(const at::Tensor& tensor,
                           const torch::lazy::BackendDevice& device) {
  xla::ComputationClient::DataPtr data = TensorToXlaData(tensor, device);
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

std::vector<xla::ComputationClient::DataPtr> Execute(
    absl::Span<const ir::Value> roots,
    const torch::lazy::BackendDevice& device) {
  ir::LoweringContext lowering_ctx("Execute", device);
  for (auto node : roots) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(node.node.get(), node.index));
    lowering_ctx.AddResult(root);
  }

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.Build());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(device.type()));

  std::vector<xla::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), device.toString(),
                       xla::ComputationClient::Get()->GetCompilationDevices(
                           device.toString(), {}),
                       &shape});

  std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
      computations =
          xla::ComputationClient::Get()->Compile(std::move(instances));

  xla::ComputationClient::ExecuteComputationOptions options;
  return xla::ComputationClient::Get()->ExecuteComputation(
      *computations.front(), lowering_ctx.GetParametersData(),
      device.toString(), options);
}

std::vector<at::Tensor> Fetch(
    absl::Span<const xla::ComputationClient::DataPtr> device_data) {
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(device_data);
  std::vector<at::Tensor> tensors;
  for (auto& literal : literals) {
    tensors.push_back(MakeTensorFromXlaLiteral(
        literal, TensorTypeFromXlaType(literal.shape().element_type())));
  }
  return tensors;
}

std::vector<at::Tensor> ExecuteAndFetch(
    absl::Span<const ir::Value> roots,
    const torch::lazy::BackendDevice& device) {
  auto results = Execute(roots, device);
  return Fetch(results);
}

void AssertBackward(const torch::Tensor& xla_output,
                    const std::vector<torch::Tensor>& xla_inputs,
                    const torch::Tensor& reference_output,
                    const std::vector<torch::Tensor>& reference_inputs) {
  torch::autograd::backward({reference_output.sum()}, {});
  torch::autograd::backward({xla_output.sum()}, {});
  ASSERT_EQ(xla_inputs.size(), reference_inputs.size());
  for (size_t i = 0; i < reference_inputs.size(); ++i) {
    AllClose(xla_inputs[i].grad(), reference_inputs[i].grad());
  }
}

void TestBackward(
    const std::vector<torch::Tensor>& inputs, const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol, double atol, int derivative_level) {
  std::vector<torch::Tensor> input_vars;
  std::vector<torch::Tensor> xinput_vars;
  std::vector<torch::Tensor> inputs_w_grad;
  std::vector<torch::Tensor> xinputs_w_grad;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const torch::Tensor& input = inputs[i];
    if (input.defined()) {
      torch::Tensor oinput =
          input.clone().detach().set_requires_grad(input.requires_grad());
      input_vars.push_back(oinput);

      torch::Tensor xinput =
          CopyToDevice(input, device, /*requires_grad=*/input.requires_grad());
      xinput_vars.push_back(xinput);
      if (input.requires_grad()) {
        inputs_w_grad.push_back(oinput);
        xinputs_w_grad.push_back(xinput);
      }
    } else {
      input_vars.emplace_back();
      xinput_vars.emplace_back();
    }
  }

  torch::Tensor output = testfn(input_vars);
  torch::Tensor xoutput = testfn(xinput_vars);
  AllClose(output, xoutput, rtol, atol);

  std::vector<torch::Tensor> outs = {output};
  std::vector<torch::Tensor> xouts = {xoutput};
  for (int d = 1; d <= derivative_level; ++d) {
    // Check grad of sum(outs) w.r.t inputs_w_grad.
    torch::Tensor sum = torch::zeros_like(outs[0]).sum();
    torch::Tensor xsum = torch::zeros_like(xouts[0]).sum();
    for (int i = 0; i < outs.size(); ++i) {
      if (outs[i].requires_grad()) {
        sum += outs[i].sum();
        xsum += xouts[i].sum();
      }
    }
    // Calculating higher order derivative requires create_graph=true
    bool create_graph = d != derivative_level;
    outs = torch::autograd::grad({sum}, inputs_w_grad, /*grad_outputs=*/{},
                                 /*retain_graph=*/c10::nullopt,
                                 /*create_graph=*/create_graph,
                                 /*allow_unused=*/true);
    xouts = torch::autograd::grad({xsum}, xinputs_w_grad, /*grad_outputs=*/{},
                                  /*retain_graph=*/c10::nullopt,
                                  /*create_graph=*/create_graph,
                                  /*allow_unused=*/true);
    for (size_t i = 0; i < outs.size(); ++i) {
      ASSERT_EQ(outs[i].defined(), xouts[i].defined());
      if (outs[i].defined()) {
        AllClose(outs[i], xouts[i], rtol, atol);
      }
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
