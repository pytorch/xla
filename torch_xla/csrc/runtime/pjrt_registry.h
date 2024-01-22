#ifndef XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_
#define XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"

namespace torch_xla {
namespace runtime {

void RegisterPjRtPlugin(
    std::string name, std::string library_path,
    absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {},
    bool init_coordinator = true);

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type);

class PjRtPlugin {
 public:
  virtual std::string library_path() = 0;

  virtual std::unordered_map<std::string, xla::PjRtValueType>
  client_create_options() = 0;

  virtual bool requires_xla_coordinator() = 0;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_INITIALIZE_PJRT_H_
