#include "tensorflow/compiler/xla/xla_client/computation_client_manager.h"
#include "tensorflow/compiler/xla/xla_client/global_data_handle_mapper.h"
#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/proxy_name.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_client.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

namespace xla {

namespace {
bool verbose = true;
}

std::recursive_mutex ComputationClientManager::computation_client_map_mtx_;
std::unordered_map<std::string, std::shared_ptr<XlaClientInfo>>
    ComputationClientManager::computation_client_info_map_;

bool ComputationClientManager::Empty() {
  std::lock_guard<std::recursive_mutex> lk(computation_client_map_mtx_);
  return computation_client_info_map_.empty();
}

std::shared_ptr<ComputationClient>
ComputationClientManager::GetComputationClient(const std::string &device,
                                               bool create) {
  std::lock_guard<std::recursive_mutex> lk(computation_client_map_mtx_);
  auto found = computation_client_info_map_.find(device);
  if (found == computation_client_info_map_.end()) {
    // We've never heard of this device
    return nullptr;
  }
  const std::shared_ptr<XlaClientInfo> &info = found->second;
  if (info->computation_client_ || !create) {
    return info->computation_client_;
  }
  // Currently just do xla via grpc, but could create a different
  // type of ComputationClient client here just as well, especially if the
  // global config code can be detahced from XrtComputationClient
  if (!info->xla_client_) {
    info->xla_client_ =
        XlaComputationClient::GetXlaClient(device, /*create=*/true);
    if (!info->xla_client_) {
      throw std::runtime_error("Failed to create xla client");
    }
  }
  info->computation_client_ = info->client_factory_->Create();
  if (!info->computation_client_) {
    XLA_ERROR() << "Client factory did not supply a "
                   "working parameterless Create function";
  }
  return info->computation_client_;
}

void ComputationClientManager::PrepareToExit() {
  std::lock_guard<std::recursive_mutex> lk(computation_client_map_mtx_);
  for (auto &item : computation_client_info_map_) {
    if (item.second->computation_client_) {
      item.second->computation_client_->PrepareToExit();
    }
  }
}

void ComputationClientManager::SetDeviceFactory(
    const std::string &device, const std::string &address,
    const std::shared_ptr<ComputationClientFactory> client_factory) {
  if (device.empty()) {
    throw std::runtime_error("Invalid empty device string");
  }
  assert(!ProxyName::is_proxy_device_name(device));
  std::shared_ptr<xla::ServiceInterface>
      old_client; // if exists, to be destroyed out of lock scope
  std::lock_guard<std::recursive_mutex> lk(computation_client_map_mtx_);
  if (verbose) {
    std::cout << "Setting device proxy: " << device << " -> " << address
              << ", proxy will have device name: "
              << ProxyName::proxy_device_name(device) << std::endl;
  }
  const std::string proxy_device_name = ProxyName::proxy_device_name(device);
  if (address.empty()) {
    // remove it
    computation_client_info_map_.erase(proxy_device_name);
    return;
  }
  auto iter = computation_client_info_map_.find(proxy_device_name);
  if (iter == computation_client_info_map_.end()) {
    auto new_info = std::make_shared<XlaClientInfo>();
    iter = computation_client_info_map_
               .emplace(std::make_pair(proxy_device_name, new_info))
               .first;
    new_info->address_ = address;
    new_info->client_factory_ = client_factory;
  } else {
    // was already there
    if (iter->second->address_ != address) {
      // If it changed, kill the old one (if it was created at all)
      old_client =
          iter->second->xla_client_; // keep a ref until out of lock scope
      iter->second->xla_client_.reset();
      iter->second->address_ = address;
      iter->second->client_factory_ = client_factory;
    }
  }
}

} // namespace xla
