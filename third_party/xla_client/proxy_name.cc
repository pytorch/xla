#include "tensorflow/compiler/xla/xla_client/proxy_name.h"
#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_computation_client.h"

namespace xla {

namespace {

const std::string PROXYABLE_DEVICE_PREFIX = "WSE:";
constexpr char PROXYABLE_DEVICE_SUFFIX = 'P';

} // namespace

bool ProxyName::is_proxy_device_name(const std::string &device) {
  if (!ProxyComputationClient::IsEnabled())
    return false;
  std::vector<std::string> parts = split(device, ':');
  assert(parts.size() == 2);
  const std::string &dev = parts[0];
  assert(!dev.empty());
  return dev.at(dev.size() - 1) == PROXYABLE_DEVICE_SUFFIX;
}

std::string ProxyName::unproxy_device_name(const std::string &device) {
  assert(ProxyComputationClient::IsEnabled());
  std::vector<std::string> parts = split(device, ':');
  assert(parts.size() == 2);
  std::string &dev = parts[0];
  assert(!dev.empty());
  assert(dev.at(dev.size() - 1) == PROXYABLE_DEVICE_SUFFIX);
  dev.resize(dev.size() - 1);
  assert(!dev.empty());
  assert(dev.at(dev.size() - 1) != PROXYABLE_DEVICE_SUFFIX);
  std::stringstream ss;
  ss << dev << ':' << parts[1];
  return ss.str();
}

std::string ProxyName::proxy_device_name(const std::string &device) {
  assert(ProxyComputationClient::IsEnabled());
  std::vector<std::string> parts = split(device, ':');
  assert(parts.size() == 2);
  const std::string &dev = parts[0];
  assert(!dev.empty());
  assert(dev.at(dev.size() - 1) != PROXYABLE_DEVICE_SUFFIX);
  std::stringstream ss;
  ss << dev << PROXYABLE_DEVICE_SUFFIX << ":" << parts[1];
  return ss.str();
}

bool ProxyName::is_proxyable_device(const std::string device) {
  if (!ProxyComputationClient::IsEnabled())
    return false;
  return strncmp(device.c_str(), PROXYABLE_DEVICE_PREFIX.c_str(),
                 PROXYABLE_DEVICE_PREFIX.size()) == 0;
}

} // namespace xla
