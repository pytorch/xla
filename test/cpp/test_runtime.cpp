#include "torch_xla/csrc/runtime/runtime.h"

#include <gtest/gtest.h>

namespace torch_xla::runtime {

TEST(RuntimeTest, NullComputationClient) {
  auto* client = GetComputationClientIfInitialized();
  EXPECT_EQ(client, nullptr);
}

TEST(RuntimeTest, GetComputationClientSuccess) {
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
