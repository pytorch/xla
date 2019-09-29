#pragma once

#include <gtest/gtest.h>

namespace torch_xla {
namespace cpp_test {

class XlaTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup();
};

class TorchXlaTest : public XlaTest {
 protected:
  static void SetUpTestCase();
};

class AtenXlaTensorTestBase : public XlaTest {
 protected:
  static void SetUpTestCase();
};

}  // namespace cpp_test
}  // namespace torch_xla
