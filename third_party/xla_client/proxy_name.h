#pragma once

#include <string>

namespace xla {

class ProxyName {
 public:
  static bool is_proxy_device_name(const std::string &device);

  static std::string unproxy_device_name(const std::string &device);

  static std::string proxy_device_name(const std::string &device);

  static bool is_proxyable_device(const std::string device);
};

}  // namespace xla
