#include "torch_xla/csrc/runtime/sys_util.h"

#include <gtest/gtest.h>

namespace torch_xla {
namespace runtime {
namespace sys_util {

TEST(SysUtilTest, Env) {
  EXPECT_EQ(GetEnvInt("does-not-exist-hopefully", 42), 42);
  EXPECT_EQ(GetEnvString("does-not-exist-hopefully", "42"), "42");
  EXPECT_EQ(GetEnvDouble("does-not-exist-hopefully", 42.0f), 42.0f);

  setenv("ordinal", "42", true);
  EXPECT_EQ(GetEnvOrdinalPath("does-not-exist-hopefully", "/path/to/test/data",
                              "ordinal"),
            "/path/to/test/data.42");

  EXPECT_EQ(GetEnvBool("does-not-exist-hopefully", true), true);
  setenv("existing-bool", "true", true);
  EXPECT_EQ(GetEnvBool("existing-bool", false), true);
  setenv("existing-bool", "false", true);
  EXPECT_EQ(GetEnvBool("existing-bool", true), false);

  setenv("existing-bool", "0", true);
  EXPECT_EQ(GetEnvBool("existing-bool", true), false);
  setenv("existing-bool", "7", true);
  EXPECT_EQ(GetEnvBool("existing-bool", false), true);

  EXPECT_GT(NowNs(), 0);
}

}  // namespace sys_util
}  // namespace runtime
}  // namespace torch_xla
