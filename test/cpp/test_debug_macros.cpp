#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test/cpp/cpp_test_util.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"

namespace torch_xla::cpp_test {
namespace {

// Prefix of the C++ stacktrace PyTorch adds to the error message.
constexpr char kTorchCppStacktracePrefix[] =
    "Exception raised from operator& at torch_xla/csrc/runtime/tf_logging.cpp:";

TEST(DebugMacrosTest, Check) {
  int32_t line;
  try {
    line = __LINE__ + 1;
    XLA_CHECK(false) << "Error message.";
  } catch (const c10::Error& error) {
    EXPECT_THAT(error.what(),
                testing::StartsWith(absl::StrCat(
                    "Check failed: false: Error message. (at ", __FILE__, ":",
                    line, ")\n\n", kTorchCppStacktracePrefix)));
  }
}

#define TEST_XLA_CHECK_OP_(opstr, lhs, rhs, compstr, valstr)                \
  TEST(DebugMacrosTest, Check##opstr) {                                     \
    try {                                                                   \
      XLA_CHECK_##opstr(lhs, rhs) << " Error message.";                     \
    } catch (const c10::Error& error) {                                     \
      EXPECT_THAT(                                                          \
          error.what(),                                                     \
          ::testing::StartsWith(absl::StrCat(                               \
              "Check failed: " compstr " (" valstr ") Error message. (at ", \
              __FILE__, ":", __LINE__, ")\n\n",                             \
              ::torch_xla::cpp_test::kTorchCppStacktracePrefix)));          \
    }                                                                       \
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

static void SetUp() {
  setenv("TORCH_SHOW_CPP_STACKTRACES", /* value= */ "1", /* replace= */ 1);
}

}  // namespace
}  // namespace torch_xla::cpp_test

int main(int argc, char** argv) {
  ::torch_xla::cpp_test::SetUp();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
