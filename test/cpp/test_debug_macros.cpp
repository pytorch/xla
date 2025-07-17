#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"

namespace torch_xla {
namespace {

using absl::StrCat;

TEST(DebugMacrosTest, Check) {
  auto line = __LINE__ + 1;
  EXPECT_THAT([] { XLA_CHECK(false) << "Error message."; },
              testing::ThrowsMessage<std::runtime_error>(testing::StartsWith(
                  StrCat("Check failed: false: Error message. (at ", __FILE__,
                         ":", line, ")\n*** Begin stack trace ***"))));
}

#define TEST_XLA_CHECK_OP_(opstr, lhs, rhs, compstr, valstr)                   \
  TEST(DebugMacrosTest, Check##opstr) {                                        \
    EXPECT_THAT(                                                               \
        [] { XLA_CHECK_##opstr(lhs, rhs) << " Error message."; },              \
        testing::ThrowsMessage<std::runtime_error>(testing::StartsWith(StrCat( \
            "Check failed: " compstr " (" valstr ") Error message. (at ",      \
            __FILE__, ":", __LINE__, ")\n*** Begin stack trace ***"))));       \
  }

#define TEST_XLA_CHECK_OP(opstr, op, lhs, rhs) \
  TEST_XLA_CHECK_OP_(opstr, lhs, rhs, #lhs " " #op " " #rhs, #lhs " vs. " #rhs)

TEST_XLA_CHECK_OP(EQ, ==, 5, 8)
TEST_XLA_CHECK_OP(NE, !=, 5, 5)
TEST_XLA_CHECK_OP(LE, <=, 5, 1)
TEST_XLA_CHECK_OP(LT, <, 5, 1)

// Since the implementation of GE and GT checks use their corresponding
// less-than LE and LT versions with their arguments swapped, we need to modify
// the expected error message accordingly.
//
// In other words:
//
//     XLA_CHECK_GE(5, 8)
//
// Errors with the following message:
//
//     Check failed: 8 <= 5 (8 vs. 5)
#define TEST_XLA_CHECK_GREATER(opstr, lessop, lhs, rhs)          \
  TEST_XLA_CHECK_OP_(opstr, lhs, rhs, #rhs " " #lessop " " #lhs, \
                     #rhs " vs. " #lhs)

TEST_XLA_CHECK_GREATER(GE, <=, 5, 8)
TEST_XLA_CHECK_GREATER(GT, <, 5, 8)

}  // namespace
}  // namespace torch_xla

static void SetUp() {
  setenv("TORCH_SHOW_CPP_STACKTRACES", /* value= */ "1", /* replace= */ 1);
}

int main(int argc, char** argv) {
  SetUp();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
