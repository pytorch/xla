#include "torch_xla_test.h"

#include <ATen/ATen.h>

#include "torch_xla/csrc/aten_xla_type_instances.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace cpp_test {

void TorchXlaTest::SetUp() { at::manual_seed(42); }

void AtenXlaTensorTestBase::SetUpTestCase() {
  AtenXlaType::InitializeAtenBindings();
  XlaHelpers::set_mat_mul_precision(xla::PrecisionConfig::HIGHEST);
}

}  // namespace cpp_test
}  // namespace torch_xla
