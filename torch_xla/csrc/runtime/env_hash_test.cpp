#include "torch_xla/csrc/runtime/env_hash.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>

namespace torch_xla {
namespace runtime {
namespace hash {

TEST(HashTest, CompilationEnvHashTest) {
  for (const char* flag_var : {"XLA_FLAGS", "LIBTPU_INIT_ARGS"}) {
    torch::lazy::hash_t base_hash = HashXlaEnvVars();

    // Add an ignored XLA flag to the environment
    setenv(flag_var, "--xla_dump_to=/foo/bar", /*overwrite=*/true);
    EXPECT_TRUE(base_hash == HashXlaEnvVars());

    // Add some non-ignored XLA flag to the environment
    setenv(flag_var, "--xla_foo_bar=1 --xla_bar_baz=0", /*overwrite=*/true);
    torch::lazy::hash_t nonignored_xla_flag = HashXlaEnvVars();
    EXPECT_TRUE(base_hash != nonignored_xla_flag);

    // Add an ignored XLA flag in addition to the non-ignored
    setenv(flag_var, "--xla_foo_bar=1 --xla_bar_baz=0 --xla_dump_to=/foo/bar",
           /*overwrite=*/true);
    torch::lazy::hash_t mixed_xla_flag = HashXlaEnvVars();
    EXPECT_TRUE(nonignored_xla_flag == mixed_xla_flag);

    // Reordering the XLA flags should not impact the hash
    setenv(flag_var, "--xla_bar_baz=0 --xla_dump_to=/foo/bar --xla_foo_bar=1",
           /*overwrite=*/true);
    torch::lazy::hash_t mixed_reordered_xla_flag = HashXlaEnvVars();
    EXPECT_TRUE(mixed_xla_flag == mixed_reordered_xla_flag);

    // Changing the XLA flag value should impact the hash
    setenv(flag_var, "--xla_bar_baz=1 --xla_dump_to=/foo/bar --xla_foo_bar=1",
           /*overwrite=*/true);
    torch::lazy::hash_t new_value_xla_flag = HashXlaEnvVars();
    EXPECT_TRUE(mixed_reordered_xla_flag != new_value_xla_flag);
  }

  // Modifying the value of TPU_MEGACORE should impact the hash
  torch::lazy::hash_t base_hash = HashXlaEnvVars();
  setenv("TPU_MEGACORE", "megacore", /*overwrite=*/true);
  EXPECT_TRUE(base_hash != HashXlaEnvVars());
}

}  // namespace hash
}  // namespace runtime
}  // namespace torch_xla
