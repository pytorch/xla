#ifndef CEREBRASPYTORCH_WSE_GRPC_CLIENT_HH
#define CEREBRASPYTORCH_WSE_GRPC_CLIENT_HH

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

class XlaClientInfo;
class ServiceInterface;
class ComputationClient;
class ComputationClientFactory;

class XlaClientInfo {
 public:
  inline std::shared_ptr<xla::ServiceInterface> operator()() {
    return xla_client_;
  }

  inline std::shared_ptr<xla::ServiceInterface> operator()() const {
    return xla_client_;
  }

  // Address of the service.  Formatting denotes type
  // (i.e. grpc:// for remote session)
  std::string address_;
  std::shared_ptr<ComputationClientFactory> client_factory_;
  std::shared_ptr<xla::ServiceInterface> xla_client_;  // deprecated
  std::shared_ptr<ComputationClient> computation_client_;
  std::unordered_map<int, xla::DeviceHandle> device_handles_;
};

class ComputationClientManager {
 public:
  ComputationClientManager() = default;

  static bool Empty();

  std::shared_ptr<ComputationClient> GetComputationClient(
      const std::string &device, bool create = true);

  void PrepareToExit();

  /**
   * @brief Setting device proxy addresses can be done before the object is
   * created and managed a global mapping.
   * @param device
   * @param proxy_address
   */
  static void SetDeviceFactory(
      const std::string &device, const std::string &address,
      const std::shared_ptr<xla::ComputationClientFactory> client_factory);

 private:
  friend class XlaComputationClient;  // Temporary due to separation of code

  static std::recursive_mutex computation_client_map_mtx_;
  static std::unordered_map<std::string, std::shared_ptr<XlaClientInfo>>
      computation_client_info_map_;
};

}  // namespace xla

#endif  // CEREBRASPYTORCH_WSE_GRPC_CLIENT_HH
