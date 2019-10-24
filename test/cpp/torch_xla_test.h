#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "metrics_snapshot.h"

namespace torch_xla {
namespace cpp_test {

class XlaTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup();

  bool CounterChanged(const std::string& counter_regex,
                      const std::unordered_set<std::string>* ignore_set);

 private:
  void MakeEndSnapshot();

  std::unique_ptr<MetricsSnapshot> start_msnap_;
  std::unique_ptr<MetricsSnapshot> end_msnap_;
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
