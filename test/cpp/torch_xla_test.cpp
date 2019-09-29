#include "torch_xla_test.h"

#include <ATen/ATen.h>

#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace cpp_test {

void XlaTest::SetUp() { at::manual_seed(42); }

void XlaTest::TearDown() {}

void XlaTest::CommonSetup() {
  XlaHelpers::set_mat_mul_precision(xla::PrecisionConfig::HIGHEST);
}

void TorchXlaTest::SetUpTestCase() { CommonSetup(); }

void AtenXlaTensorTestBase::SetUpTestCase() {
  CommonSetup();
  AtenXlaType::InitializeAtenBindings();
}

}  // namespace cpp_test
}  // namespace torch_xla
