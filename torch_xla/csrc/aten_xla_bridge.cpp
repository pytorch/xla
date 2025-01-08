#include "torch_xla/csrc/aten_xla_bridge.h"

#include <ATen/FunctionalTensorWrapper.h>
#include <torch/csrc/lazy/core/tensor_util.h>

#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {
namespace bridge {
namespace {

thread_local absl::optional<torch::lazy::BackendDevice> g_current_device;

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

  void SetVirtualDevice() {
    for (auto& device : GetAllDevices()) {
      if (static_cast<XlaDeviceType>(device.type()) == XlaDeviceType::SPMD) {
        return;
      }
    }
    devices_.emplace_back(ParseDeviceString("SPMD:0"));
    devices_ordinals_[devices_.back()] = 0;
  }

 private:
  AtenXlaDeviceMapper() {
    if (UseVirtualDevice()) {
      devices_.emplace_back(ParseDeviceString("SPMD:0"));
      devices_ordinals_[devices_.back()] = 0;
    } else {
      for (auto& device_str :
           torch_xla::runtime::GetComputationClient()->GetLocalDevices()) {
        devices_.emplace_back(ParseDeviceString(device_str));
        devices_ordinals_[devices_.back()] = devices_.size() - 1;
      }
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
  if (tensor.defined() &&
      at::functionalization::impl::isFunctionalTensor(tensor)) {
    // To make sure we have the most updated version of tensor.
    at::functionalization::impl::sync(tensor);
  }
  XLATensorImpl* impl = GetXlaTensorImpl(tensor);
  if (impl == nullptr) {
    return XLATensorPtr();
  }
  return impl->tensor();
}

std::vector<XLATensorPtr> TryGetXlaTensors(const at::ITensorListRef& tensors) {
  std::vector<XLATensorPtr> xla_tensors;
  xla_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    xla_tensors.push_back(bridge::TryGetXlaTensor(tensor));
  }
  return xla_tensors;
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

void ReplaceXlaTensor(const std::vector<at::Tensor>& tensors,
                      const std::vector<XLATensorPtr> new_xla_tensors) {
  XLA_CHECK(tensors.size() == new_xla_tensors.size())
      << "The size of tensors and new_xla_tensors are not equal: "
      << tensors.size() << " vs. " << new_xla_tensors.size();
  for (size_t i = 0; i < tensors.size(); ++i) {
    ReplaceXlaTensor(tensors[i], new_xla_tensors[i]);
  }
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

XLATensorPtr GetOrCreateXlaTensor(const std::optional<at::Tensor>& tensor,
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
    if (!tensor.defined()) {
      continue;
    }
    auto inner_tensor = torch::lazy::maybe_unwrap_functional(tensor);
    if (!inner_tensor.defined()) {
      continue;
    }

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

std::vector<std::optional<at::Tensor>> XlaCreateOptTensorList(
    const std::vector<std::optional<at::Tensor>>& tensors) {
  std::vector<std::optional<at::Tensor>> opt_aten_xla_tensors(tensors.size());
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

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(tensor);
  if (!xtensor) {
    return std::nullopt;
  }
  return xtensor->GetDevice();
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return GetXlaDevice(*tensor);
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return std::nullopt;
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return std::nullopt;
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorOptions& tensor_options) {
  if (!tensor_options.has_device()) {
    return std::nullopt;
  }
  return GetXlaDevice(tensor_options.device());
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::Device& device) {
  if (device.type() != at::kXLA) {
    return std::nullopt;
  }
  return AtenDeviceToXlaDevice(device);
}

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::optional<c10::Device>& device) {
  if (!device) {
    return std::nullopt;
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
  // TODO(yeounoh) until we expose SPMD virtual device to the frontend, this
  // will just be `XLA:0`.
  if (device.type() == (int8_t)XlaDeviceType::SPMD) {
    return c10::Device(at::kXLA, (size_t)0);
  }
  return c10::Device(at::kXLA,
                     AtenXlaDeviceMapper::Get()->GetDeviceOrdinal(device));
}

std::string ToXlaString(const c10::Device& device) {
  return absl::StrCat("xla:", device.index());
}

const torch::lazy::BackendDevice* GetDefaultDevice() {
  static std::string default_device_spec =
      UseVirtualDevice() ? "SPMD:0"
                         : runtime::GetComputationClient()->GetDefaultDevice();
  XLA_CHECK(!default_device_spec.empty());
  static const torch::lazy::BackendDevice default_device =
      ParseDeviceString(default_device_spec);
  return &default_device;
}

c10::Device AtenDefaultDevice() {
  return XlaDeviceToAtenDevice(*GetDefaultDevice());
}

torch::lazy::BackendDevice GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

c10::Device GetCurrentAtenDevice() {
  return XlaDeviceToAtenDevice(GetCurrentDevice());
}

c10::Device SetCurrentDevice(const c10::Device& device) {
  torch::lazy::BackendDevice prev_device =
      SetCurrentDevice(AtenDeviceToXlaDevice(device));
  return XlaDeviceToAtenDevice(prev_device);
}

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device) {
  torch::lazy::BackendDevice current = GetCurrentDevice();
  g_current_device = device;
  TF_VLOG(2) << "New current device: " << device;
  return current;
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

at::Tensor AtenFromXlaTensor(XLATensorPtr xla_tensor,
                             bool skip_functionalization) {
  if (xla_tensor) {
    auto out =
        at::Tensor(c10::make_intrusive<XLATensorImpl>(std::move(xla_tensor)));
    // See Note [Lazy Tensor Functionalization]
    if (skip_functionalization ||
        c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::Functionalize)) {
      // Invariant: if the functionalization key is in the exclude set, then
      // we're expected to return an ordinary tensor, which will be "lifted"
      // into a functional wrapper later.
      return out;
    } else {
      auto wrapped = MaybeWrapTensorToFunctional(out);
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
    const std::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    XLATensorPtr xla_tensor = XLATensor::Create(std::move(tensor), *device);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors,
    const std::optional<torch::lazy::BackendDevice>& device) {
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

const at::Tensor& GetRootBase(const at::Tensor& tensor) {
  auto xla_tensor = TryGetXlaTensor(tensor);
  if (xla_tensor && xla_tensor->Base().defined()) {
    return GetRootBase(xla_tensor->Base());
  } else {
    return tensor;
  }
}

XLATensorPtr SetBaseTensor(XLATensorPtr tensor, const at::Tensor& base) {
  XLA_CHECK(base.device().is_xla())
      << "base tensor on unexpected device: " << base.device();
  tensor->SetBase(GetRootBase(base));
  return tensor;
}

}  // namespace bridge
}  // namespace torch_xla
