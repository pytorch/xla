#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace {

constexpr char new_message[] = "New test error message";
constexpr char message[] = "Test error message";
constexpr char test_file[] = "test_file.cpp";
constexpr int32_t line = 42;

TEST(StatusWithErrorContextTest, MaybeWithLocationRetunsSameStatus) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result = MaybeWithLocation(error_status, test_file, line);
  ASSERT_NE(result, error_status);
  ASSERT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), "Test error message (at test_file.cpp:42)");
}

TEST(StatusWithErrorContextTest, MaybeWithNewMessageEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result = MaybeWithNewMessage(error_status, test_file, line);
  ASSERT_NE(result, error_status);
  ASSERT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), "Test error message (at test_file.cpp:42)");
}

TEST(StatusWithErrorContextTest, MaybeWithNewMessageNonEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result =
      MaybeWithNewMessage(error_status, test_file, line, new_message);
  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(),
            "New test error message (at test_file.cpp:42)\n"
            "From Error: Test error message");
}

TEST(StatusWithErrorContextTest, MacroReturnIfErrorWithError) {
  int32_t err_line = 0;

  auto test_function = [=, &err_line]() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(message);
    err_line = __LINE__ + 1;
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), absl::StrCat("Test error message (at ", __FILE__,
                                           ":", err_line, ")"));
}

TEST(StatusWithErrorContextTest, MacroAssignOrReturnWithError) {
  int32_t err_line = 0;

  auto test_function = [&err_line]() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError(message);
    err_line = __LINE__ + 1;
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      result.status().message(),
      absl::StrCat("Test error message (at ", __FILE__, ":", err_line, ")"));
}

TEST(StatusWithErrorContextTest, MacroErrorWithLocation) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  int32_t err_line = __LINE__ + 1;
  absl::Status result = XLA_ERROR_WITH_LOCATION(error_status);
  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), absl::StrCat("Test error message (at ", __FILE__,
                                           ":", err_line, ")"));
}

void SetUp() {
  setenv(runtime::env::kEnvShowCppErrorContext, /* value= */ "true",
         /* replace= */ 1);
}

}  // namespace
}  // namespace torch_xla

int main(int argc, char** argv) {
  ::torch_xla::SetUp();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
