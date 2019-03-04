#pragma once

#include <gtest/gtest.h>

namespace torch_xla {
namespace cpp_test {

class TorchXlaTest : public ::testing::Test {
 protected:
  void SetUp() override;
};

class AtenXlaTensorTestBase : public TorchXlaTest {
 protected:
  static void SetUpTestCase();
};

}  // namespace cpp_test
}  // namespace torch_xla
