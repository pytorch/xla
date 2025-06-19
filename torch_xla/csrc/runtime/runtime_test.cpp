#include "torch_xla/csrc/runtime/runtime.h"

#include <gtest/gtest.h>

namespace torch_xla::runtime {

TEST(RuntimeTest, NullComputationClient) {
  auto client = GetComputationClientIfInitialized();
  EXPECT_EQ(client, nullptr);
}

TEST(RuntimeTest, GetComputationClientSuccess) {
  ComputationClient* client;

  client = GetComputationClientIfInitialized();
  EXPECT_EQ(client, nullptr);

  // Initialize the ComputationClient.
  // Check all the APIs return the same valid ComputationClient.

  client = GetComputationClientOrDie();
  EXPECT_NE(client, nullptr);

  auto status = GetComputationClient();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(client, status.value());

  EXPECT_EQ(client, GetComputationClientIfInitialized());
}

}  // namespace torch_xla::runtime
