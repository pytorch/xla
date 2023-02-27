#include "torch_xla/csrc/aten_xla_bridge.h"

#include <ATen/FunctionalTensorWrapper.h>
#include <torch/csrc/lazy/core/tensor_util.h>

#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {
namespace bridge {
namespace {

class AtenXlaDeviceMapper {
 public:
  static AtenXlaDeviceMapper* Get();

  size_t GetDeviceOrdinal(const torch::lazy::BackendDevice& device) const {
    auto it = devices_ordinals_.find(device);
    XLA_CHECK(it != devices_ordinals_.end()) << device;
    return it->second;
  }

  const torch::lazy::BackendDevice& GetDeviceFromOrdinal(size_t ordinal) const {
    return devices_.at(ordinal);
  }

  std::vector<torch::lazy::BackendDevice> GetAllDevices() const {
    return devices_;
  }

 private:
  AtenXlaDeviceMapper() {
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      devices_.emplace_back(ParseDeviceString(device_str));
      devices_ordinals_[devices_.back()] = devices_.size() - 1;
    }
  }

  std::vector<torch::lazy::BackendDevice> devices_;
  std::map<torch::lazy::BackendDevice, size_t> devices_ordinals_;
};

AtenXlaDeviceMapper* AtenXlaDeviceMapper::Get() {
  static AtenXlaDeviceMapper* device_mapper = new AtenXlaDeviceMapper();
  return device_mapper;
}

XLATensorImpl* GetXlaTensorImpl(const at::Tensor& tensor) {
  auto inner_tensor = torch::lazy::maybe_unwrap_functional(tensor);
  return dynamic_cast<XLATensorImpl*>(inner_tensor.unsafeGetTensorImpl());
}

}  // namespace

XLATensorPtr TryGetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl = GetXlaTensorImpl(tensor);
  if (impl == nullptr) {
    return XLATensorPtr();
  }
  return impl->tensor();
}

bool IsXlaTensor(const at::Tensor& tensor) {
  return GetXlaTensorImpl(tensor) != nullptr;
}

XLATensorPtr GetXlaTensor(const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(tensor);
  XLA_CHECK(xtensor) << "Input tensor is not an XLA tensor: "
                     << tensor.toString();
  return xtensor;
}

void ReplaceXlaTensor(const at::Tensor& tensor, XLATensorPtr new_xla_tensor) {
  auto inner_tensor = torch::lazy::maybe_unwrap_functional(tensor);
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(inner_tensor.unsafeGetTensorImpl());
  XLA_CHECK(impl != nullptr)
      << "Input tensor is not an XLA tensor: " << inner_tensor.toString();
  impl->set_tensor(std::move(new_xla_tensor));
}

std::vector<XLATensorPtr> GetXlaTensors(const at::ITensorListRef& tensors) {
  std::vector<XLATensorPtr> xla_tensors;
  xla_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    xla_tensors.push_back(bridge::GetXlaTensor(tensor));
  }
  return xla_tensors;
}

torch_xla::XLATensorPtr GetXlaTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number() ||
      (tensor.dim() == 0 && tensor.numel() == 1)) {
    return torch_xla::bridge::GetOrCreateXlaTensor(tensor, device);
  } else {
    return torch_xla::bridge::GetXlaTensor(tensor);
  }
}

XLATensorPtr GetOrCreateXlaTensor(const at::Tensor& tensor,
                                  const torch::lazy::BackendDevice& device) {
  if (!tensor.defined()) {
    return XLATensorPtr();
  }
  auto inner_tensor = torch::lazy::maybe_unwrap_functional(tensor);
  if (!inner_tensor.defined()) {
    return XLATensorPtr();
  }
  auto xtensor = TryGetXlaTensor(tensor);
  return xtensor ? xtensor : XLATensor::Create(inner_tensor, device);
}

XLATensorPtr GetOrCreateXlaTensor(const c10::optional<at::Tensor>& tensor,
                                  const torch::lazy::BackendDevice& device) {
  if (!IsDefined(tensor)) {
    return XLATensorPtr();
  }
  auto xtensor = TryGetXlaTensor(*tensor);
  auto inner_tensor = torch::lazy::maybe_unwrap_functional(*tensor);
  return xtensor ? xtensor : XLATensor::Create(inner_tensor, device);
}

std::vector<XLATensorPtr> GetOrCreateXlaTensors(
    absl::Span<const at::Tensor> tensors,
    const torch::lazy::BackendDevice& device) {
  std::vector<XLATensorPtr> xla_tensors;
  for (const at::Tensor& tensor : tensors) {
    xla_tensors.push_back(bridge::GetOrCreateXlaTensor(tensor, device));
  }
  return xla_tensors;
}

std::vector<at::Tensor> XlaCreateTensorList(const at::ITensorListRef& tensors) {
  std::vector<at::Tensor> aten_xla_tensors(tensors.size());
  std::vector<XLATensorPtr> xla_tensors;
  // We need to separate out the defined tensors first, GetXlaTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> to_translate(tensors.size());
  size_t ix = 0;
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) continue;
    auto inner_tensor = torch::lazy::maybe_unwrap_functional(tensor);
    if (!inner_tensor.defined()) continue;

    auto xtensor = TryGetXlaTensor(tensor);
    if (xtensor) {
      to_translate[ix] = true;
      xla_tensors.push_back(xtensor);
    } else {
      aten_xla_tensors[ix] = tensor;
    }
    ++ix;
  }
  auto defined_aten_xla_tensors =
      XLAGraphExecutor::Get()->GetTensors(&xla_tensors);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      auto tensor = defined_aten_xla_tensors[defined_pos++];
      XLA_CHECK(!at::functionalization::impl::isFunctionalTensor(tensor))
          << "Expected non-functional tensor!";
      // This function is responsible for returning CPU tensors.
      // So we do not want to wrap the outputs into FunctionalTensorWrappers.
      aten_xla_tensors[i] = tensor;
    }
  }
  return aten_xla_tensors;
}

std::vector<c10::optional<at::Tensor>> XlaCreateOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> opt_aten_xla_tensors(tensors.size());
  std::vector<at::Tensor> materialized_tensors;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto tensor = tensors[i];
    if (tensor.has_value()) {
      to_translate[i] = true;
      materialized_tensors.push_back(*tensor);
    }
  }
  auto aten_materialzied_tensors = XlaCreateTensorList(materialized_tensors);
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      opt_aten_xla_tensors[i] =
          std::move(aten_materialzied_tensors[defined_pos++]);
    }
  }
  return opt_aten_xla_tensors;
}

void XlaUpdateTensors(absl::Span<const at::Tensor> dest_xla_tensors,
                      absl::Span<const at::Tensor> source_cpu_tensors,
                      absl::Span<const size_t> indices) {
  for (auto index : indices) {
    at::Tensor dest = dest_xla_tensors.at(index);
    at::Tensor source = source_cpu_tensors.at(index);
    XLATensorImpl* dest_impl = GetXlaTensorImpl(dest);
    if (dest_impl != nullptr) {
      auto xla_source = TryGetXlaTensor(source);
      if (!xla_source) {
        dest_impl->tensor()->UpdateFromTensorOut(source);
      } else {
        dest_impl->tensor()->UpdateFromTensorOut(xla_source);
      }
      dest_impl->force_refresh_sizes();
    } else {
      dest.resize_as_(source).copy_(source);
    }
  }
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(tensor);
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return GetXlaDevice(*tensor);
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorOptions& tensor_options) {
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetXlaDevice(tensor_options.device());
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::Device& device) {
  if (device.type() != at::kXLA) {
    return c10::nullopt;
  }
  return AtenDeviceToXlaDevice(device);
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  return GetXlaDevice(*device);
}

std::vector<torch::lazy::BackendDevice> GetBackendDevices() {
  return AtenXlaDeviceMapper::Get()->GetAllDevices();
}

torch::lazy::BackendDevice AtenDeviceToXlaDevice(const c10::Device& device) {
  XLA_CHECK_EQ(device.type(), at::kXLA) << device;
  int ordinal = device.has_index() ? device.index() : -1;
  if (ordinal < 0) {
    c10::Device current_device = GetCurrentAtenDevice();
    if (current_device.has_index()) {
      ordinal = current_device.index();
    }
  }
  if (ordinal < 0) {
    return GetCurrentDevice();
  }
  return AtenXlaDeviceMapper::Get()->GetDeviceFromOrdinal(ordinal);
}

c10::Device XlaDeviceToAtenDevice(const torch::lazy::BackendDevice& device) {
  return c10::Device(at::kXLA,
                     AtenXlaDeviceMapper::Get()->GetDeviceOrdinal(device));
}

std::string ToXlaString(const c10::Device& device) {
  return absl::StrCat("xla:", device.index());
}

c10::Device AtenDefaultDevice() {
  return XlaDeviceToAtenDevice(*GetDefaultDevice());
}

c10::Device SetCurrentDevice(const c10::Device& device) {
  torch::lazy::BackendDevice prev_device =
      torch_xla::SetCurrentDevice(AtenDeviceToXlaDevice(device));
  return XlaDeviceToAtenDevice(prev_device);
}

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device) {
  return torch_xla::SetCurrentDevice(device);
}

c10::Device GetCurrentAtenDevice() {
  return XlaDeviceToAtenDevice(torch_xla::GetCurrentDevice());
}

at::Tensor XlaToAtenTensor(XLATensorPtr xla_tensor,
                           const at::TensorOptions& tensor_options) {
  if (tensor_options.has_device()) {
    XLA_CHECK_NE(tensor_options.device().type(), at::kXLA);
  }
  at::Tensor tensor = xla_tensor->ToTensor(/*detached=*/false);
  // We need to copy the tensor since it is cached within the XLATensor, and
  // returning it directly might expose it to in place changes. Which there was
  // COW option :)
  return tensor.to(tensor_options, /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor AtenFromXlaTensor(XLATensorPtr xla_tensor) {
  if (xla_tensor) {
    auto out =
        at::Tensor(c10::make_intrusive<XLATensorImpl>(std::move(xla_tensor)));
    // See Note [Lazy Tensor Functionalization]
    if (c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::Functionalize)) {
      // Invariant: if the functionalization key is in the exclude set, then
      // we're expected to return an ordinary tensor, which will be "lifted"
      // into a functional wrapper later.
      return out;
    } else {
      auto wrapped = at::functionalization::impl::to_functional_tensor(out);
      return wrapped;
    }
  } else {
    return at::Tensor();
  }
}

std::vector<at::Tensor> AtenFromXlaTensors(
    absl::Span<const XLATensorPtr> xla_tensors) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(xla_tensors.size());
  for (auto& tensor : xla_tensors) {
    tensors.emplace_back(AtenFromXlaTensor(tensor));
  }
  return tensors;
}

at::Tensor CreateXlaTensor(
    at::Tensor tensor,
    const c10::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    XLATensorPtr xla_tensor = XLATensor::Create(std::move(tensor), *device);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors,
    const c10::optional<torch::lazy::BackendDevice>& device) {
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

}  // namespace bridge
}  // namespace torch_xla
