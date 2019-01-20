#include "cpp_test_util.h"

#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace torch_xla {
namespace cpp_test {

void ForEachDevice(const std::function<void(const Device&)>& devfn) {
  devfn(Device("CPU:0"));

  std::string default_device =
      xla::ComputationClient::Get()->GetDefaultDevice();
  if (default_device != "CPU:0") {
    devfn(Device(default_device));
  }
}

}  // namespace cpp_test
}  // namespace torch_xla
