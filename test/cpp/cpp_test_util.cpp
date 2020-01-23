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
                           const torch::Device& device) {
  return tensor.to(device, /*non_blocking=*/false, /*copy=*/true);
}

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
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

void ForEachDevice(const std::function<void(const Device&)>& devfn) {
  const Device* device = GetDefaultDevice();
  bridge::SetCurrentDevice(*device);
  devfn(*device);
}

void ForEachDevice(const std::function<void(const torch::Device&)>& devfn) {
  torch::Device device = bridge::AtenDefaultDevice();
  bridge::SetCurrentDevice(device);
  devfn(device);
}

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol,
                 double atol) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
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
    DeviceType device_type,
    const std::function<void(const std::vector<Device>&,
                             const std::vector<Device>&)>& devfn) {
  std::vector<Device> devices;
  std::vector<Device> all_devices;
  for (const auto& device_str :
       xla::ComputationClient::Get()->GetLocalDevices()) {
    Device device(device_str);
    if (device.hw_type == device_type) {
      devices.push_back(device);
    }
  }
  for (const auto& device_str :
       xla::ComputationClient::Get()->GetAllDevices()) {
    Device device(device_str);
    if (device.hw_type == device_type) {
      all_devices.push_back(device);
    }
  }
  if (!devices.empty()) {
    devfn(devices, all_devices);
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
  return ir::DumpUtil::ToHlo({xtensor.GetIrValue()});
}

ir::Value GetTensorIrValue(const at::Tensor& tensor, const Device& device) {
  xla::ComputationClient::DataPtr data = TensorToXlaData(tensor, device);
  return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

std::vector<xla::ComputationClient::DataPtr> Execute(
    tensorflow::gtl::ArraySlice<const ir::Value> roots, const Device& device) {
  ir::LoweringContext lowering_ctx("Execute");
  for (auto node : roots) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(node);
    lowering_ctx.AddResult(root);
  }

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.Build());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());
  xla::Shape shape =
      MakeShapeWithDeviceLayout(program_shape.result(), device.hw_type);

  std::vector<xla::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), device.ToString(),
                       xla::ComputationClient::Get()->GetCompilationDevices(
                           device.ToString(), {}),
                       &shape});

  std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
      computations =
          xla::ComputationClient::Get()->Compile(std::move(instances));

  xla::ComputationClient::ExecuteComputationOptions options;
  return xla::ComputationClient::Get()->ExecuteComputation(
      *computations.front(), lowering_ctx.GetParametersData(),
      device.ToString(), options);
}

std::vector<at::Tensor> Fetch(
    tensorflow::gtl::ArraySlice<const xla::ComputationClient::DataPtr>
        device_data) {
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
    tensorflow::gtl::ArraySlice<const ir::Value> roots, const Device& device) {
  auto results = Execute(roots, device);
  return Fetch(results);
}

void TestBackward(
    const std::vector<torch::Tensor>& inputs, const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol, double atol) {
  std::vector<torch::Tensor> input_vars;
  std::vector<torch::Tensor> xinput_vars;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const torch::Tensor& input = inputs[i];
    if (input.defined()) {
      input_vars.push_back(
          input.clone().detach().set_requires_grad(input.requires_grad()));

      torch::Tensor xinput = CopyToDevice(input, device)
                                 .detach()
                                 .set_requires_grad(input.requires_grad());
      xinput_vars.push_back(xinput);
    } else {
      input_vars.emplace_back();
      xinput_vars.emplace_back();
    }
  }

  torch::Tensor output = testfn(input_vars);
  torch::Tensor xoutput = testfn(xinput_vars);
  AllClose(output, xoutput, rtol, atol);
  output.backward(torch::ones_like(output));
  xoutput.backward(torch::ones_like(xoutput));
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].defined() && inputs[i].requires_grad()) {
      ASSERT_TRUE(xinput_vars[i].grad().defined());
      AllClose(input_vars[i].grad(), xinput_vars[i].grad(), rtol, atol);
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
