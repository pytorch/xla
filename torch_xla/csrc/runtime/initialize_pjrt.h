#ifndef XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_
#define XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_

#include "xla/pjrt/pjrt_client.h"

namespace torch_xla {
namespace runtime {

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type);

}
}  // namespace torch_xla

#endif  // XLA_CLIENT_INITIALIZE_PJRT_H_
