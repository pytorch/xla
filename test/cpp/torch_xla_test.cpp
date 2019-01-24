#include "torch_xla_test.h"
#include <ATen/ATen.h>

namespace torch_xla {
namespace cpp_test {

void TorchXlaTest::SetUp() { at::manual_seed(42); }

}  // namespace cpp_test
}  // namespace torch_xla
