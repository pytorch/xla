#include "torch_xla/csrc/aten_xla_bridge.h"

#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace bridge {
namespace {

class AtenXlaDeviceMapper {
 public:
  static AtenXlaDeviceMapper* Get();

  size_t GetDeviceOrdinal(const Device& device) const {
    auto it = devices_ordinals_.find(device);
    XLA_CHECK(it != devices_ordinals_.end()) << device;
    return it->second;
  }

  const Device& GetDeviceFromOrdinal(size_t ordinal) const {
    return devices_.at(ordinal);
  }

 private:
  AtenXlaDeviceMapper() {
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      devices_.emplace_back(device_str);
      devices_ordinals_[devices_.back()] = devices_.size() - 1;
    }
  }

  std::vector<Device> devices_;
  std::map<Device, size_t> devices_ordinals_;
};

AtenXlaDeviceMapper* AtenXlaDeviceMapper::Get() {
  static AtenXlaDeviceMapper* device_mapper = new AtenXlaDeviceMapper();
  return device_mapper;
}

}  // namespace

c10::optional<XLATensor> TryGetXlaTensor(const at::Tensor& tensor) {
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return c10::nullopt;
  }
  return impl->tensor();
}

XLATensor GetXlaTensor(const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(tensor);
  XLA_CHECK(xtensor) << "Input tensor is not an XLA tensor: "
                     << tensor.toString();
  return *xtensor;
}

std::vector<XLATensor> GetXlaTensors(
    tensorflow::gtl::ArraySlice<const at::Tensor> tensors) {
  std::vector<XLATensor> xla_tensors;
  xla_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    xla_tensors.push_back(bridge::GetXlaTensor(tensor));
  }
  return xla_tensors;
}

XLATensor GetXlaTensorUnwrap(const at::Tensor& tensor) {
  return GetXlaTensor(ToTensor(tensor));
}

XLATensor GetOrCreateXlaTensor(const at::Tensor& tensor, const Device& device) {
  if (!tensor.defined()) {
    return XLATensor();
  }
  auto xtensor = TryGetXlaTensor(tensor);
  return xtensor ? *xtensor : XLATensor::Create(tensor, device);
}

std::vector<at::Tensor> XlaCreateTensorList(
    const at::TensorList& tensors, const std::vector<bool>* writeable) {
  std::vector<at::Tensor> aten_xla_tensors(tensors.size());
  std::vector<XLATensor> xla_tensors;
  // We need to separate out the defined tensors first, GetXlaTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> defined_writeable;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (!tensor.defined()) {
      XLA_CHECK(writeable == nullptr || !(*writeable)[i])
          << "Trying to write to an undefined tensor";
    } else if (tensor.device().is_cpu()) {
      aten_xla_tensors[i] = ToTensor(tensor);
    } else {
      to_translate[i] = true;
      xla_tensors.push_back(GetXlaTensorUnwrap(tensor));
      if (writeable != nullptr) {
        defined_writeable.push_back((*writeable)[i]);
      }
    }
  }
  auto defined_aten_xla_tensors = XLATensor::GetTensors(
      &xla_tensors, writeable ? &defined_writeable : nullptr);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_xla_tensors[i] = std::move(defined_aten_xla_tensors[defined_pos++]);
    }
  }
  return aten_xla_tensors;
}

c10::optional<Device> GetXlaDevice(const at::Tensor& tensor) {
  auto xtensor = TryGetXlaTensor(ToTensor(tensor));
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
}

c10::optional<Device> GetXlaDevice(const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<Device> GetXlaDevice(const at::TensorOptions& tensor_options) {
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetXlaDevice(tensor_options.device());
}

c10::optional<Device> GetXlaDevice(const c10::Device& device) {
  if (device.type() != at::kXLA) {
    return c10::nullopt;
  }
  return AtenDeviceToXlaDevice(device);
}

Device AtenDeviceToXlaDevice(const c10::Device& device) {
  XLA_CHECK_EQ(device.type(), at::kXLA) << device;
  int ordinal = device.has_index() ? device.index() : -1;
  if (ordinal < 0) {
    c10::Device current_device = XLATensorImpl::GetCurrentAtenDevice();
    if (current_device.has_index()) {
      ordinal = current_device.index();
    }
  }
  if (ordinal < 0) {
    return *GetDefaultDevice();
  }
  return AtenXlaDeviceMapper::Get()->GetDeviceFromOrdinal(ordinal);
}

c10::Device XlaDeviceToAtenDevice(const Device& device) {
  return c10::Device(at::kXLA,
                     AtenXlaDeviceMapper::Get()->GetDeviceOrdinal(device));
}

std::string ToXlaString(const c10::Device& device) {
  return absl::StrCat("xla:", device.index());
}

c10::Device AtenDefaultDevice() {
  return XlaDeviceToAtenDevice(*GetDefaultDevice());
}

at::Tensor AtenFromXlaTensor(XLATensor xla_tensor) {
  return at::Tensor(c10::make_intrusive<XLATensorImpl>(std::move(xla_tensor)));
}

std::vector<at::Tensor> AtenFromXlaTensors(
    tensorflow::gtl::ArraySlice<const XLATensor> xla_tensors) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(xla_tensors.size());
  for (auto& tensor : xla_tensors) {
    tensors.emplace_back(AtenFromXlaTensor(tensor));
  }
  return tensors;
}

at::Tensor CreateXlaTensor(at::Tensor tensor,
                           const c10::optional<Device>& device) {
  if (tensor.defined() && device) {
    XLATensor xla_tensor = XLATensor::Create(std::move(tensor), *device);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

std::vector<at::Tensor> CreateXlaTensors(const std::vector<at::Tensor>& tensors,
                                         const c10::optional<Device>& device) {
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

}  // namespace bridge
}  // namespace torch_xla
