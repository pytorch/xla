#include <gtest/gtest.h>

#include "torch_xla/csrc/runtime/runtime.h"

namespace torch_xla::runtime {

TEST(RuntimeTest, ComputationClientInitialization) {
  ComputationClient* client;

  client = GetComputationClientIfInitialized();
  EXPECT_EQ(client, nullptr);

  // Initialize the ComputationClient.
  // Check all the APIs return the same valid ComputationClient.

  client = GetComputationClientOrDie();
  ASSERT_NE(client, nullptr);

  auto status = GetComputationClient();
  ASSERT_TRUE(status.ok());

  EXPECT_EQ(status.value(), client);
  EXPECT_EQ(GetComputationClientIfInitialized(), client);
}

}  // namespace torch_xla::runtime
