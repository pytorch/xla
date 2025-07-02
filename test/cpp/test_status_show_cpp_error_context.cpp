#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

// Reminder
// ========
//
// This file is a companion to test_status_dont_show_cpp_error_context.cpp.
// This file specifically tests behavior when XLA_SHOW_CPP_ERROR_CONTEXT is
// set to "true".
//
// If you add or delete a test in this file, please make the corresponding
// change in test_status_dont_show_cpp_error_context.cpp as well, adapting
// for XLA_SHOW_CPP_ERROR_CONTEXT being "false" in that file.

namespace torch_xla {
namespace {

using absl::Status;
using absl::StatusCode;
using absl::StatusOr;
using absl::StrCat;

constexpr char new_message[] = "New test error message";
constexpr char message[] = "Test error message";
constexpr char test_file[] = "test_file.cpp";
constexpr int32_t line = 42;

TEST(StatusWithErrorContextTest, MaybeThrowWithOkStatus) {
  Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST(StatusWithErrorContextTest, MaybeThrowWithErrorStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST(StatusWithErrorContextTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST(StatusWithErrorContextTest, GetValueOrThrowWithErrorStatusOr) {
  StatusOr<int> status_or = absl::InvalidArgumentError(message);
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST(StatusWithErrorContextTest, MaybeWithLocationRetunsSameStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithLocation(error_status, test_file, line);
  ASSERT_NE(result, error_status);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), "Test error message (at test_file.cpp:42)");
}

TEST(StatusWithErrorContextTest, MaybeWithNewMessageEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithNewMessage(error_status, test_file, line);
  EXPECT_EQ(result, error_status);
}

TEST(StatusWithErrorContextTest, MaybeWithNewMessageNonEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result =
      MaybeWithNewMessage(error_status, test_file, line, new_message);

  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(),
            StrCat("New test error message (at test_file.cpp:42)\n"
                   "From Error: Test error message"));
}

TEST(StatusWithErrorContextTest, MacroReturnIfError) {
  int value = 42;

  auto test_function = [=]() -> StatusOr<int> {
    Status ok_status = absl::OkStatus();
    XLA_RETURN_IF_ERROR(ok_status);
    return value;
  };

  StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), value);
}

TEST(StatusWithErrorContextTest, MacroReturnIfErrorWithError) {
  auto test_function = [=]() -> Status {
    Status error_status = absl::InvalidArgumentError(message);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(message));
}

TEST(StatusWithErrorContextTest, MacroReturnIfErrorWithNestedError) {
  auto test_function = []() -> Status {
    Status error_status = absl::InvalidArgumentError(message);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  auto outer_test_function = [&]() -> Status {
    XLA_RETURN_IF_ERROR(test_function());
    return absl::OkStatus();
  };

  Status result = outer_test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(message));
}

TEST(StatusWithErrorContextTest, MacroReturnIfErrorWithErrorWithNewMessage) {
  int32_t errline = 0;
  auto test_function = [&errline]() -> Status {
    Status error_status = absl::InvalidArgumentError(message);
    errline = __LINE__ + 1;
    XLA_RETURN_IF_ERROR(error_status, new_message);
    return absl::OkStatus();
  };

  Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(),
            StrCat("New test error message (at ", __FILE__, ":", errline,
                   ")\nFrom Error: Test error message"));
}

TEST(StatusWithErrorContextTest, MacroAssignOrReturn) {
  int initial_value = 42;
  int expected_value = initial_value * 2;

  auto test_function = [=]() -> StatusOr<int> {
    StatusOr<int> status_or = initial_value;
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), expected_value);
}

TEST(StatusWithErrorContextTest, MacroAssignOrReturnWithError) {
  auto test_function = []() -> StatusOr<int> {
    StatusOr<int> status_or = absl::InvalidArgumentError(message);
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), std::string_view(message));
}

TEST(StatusWithErrorContextTest, MacroAssignOrReturnWithErrorWithNewMessage) {
  int32_t errline = 0;

  auto test_function = [&errline]() -> StatusOr<int> {
    StatusOr<int> status_or = absl::InvalidArgumentError(message);
    errline = __LINE__ + 1;
    XLA_ASSIGN_OR_RETURN(int value, status_or, new_message);
    return value * 2;
  };

  StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(),
            StrCat("New test error message (at ", __FILE__, ":", errline,
                   ")\nFrom Error: Test error message"));
}

TEST(StatusWithErrorContextTest, MacroErrorWithLocation) {
  Status error_status = absl::InvalidArgumentError(message);
  int32_t errline = __LINE__ + 1;
  Status result = XLA_ERROR_WITH_LOCATION(error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(),
            StrCat("Test error message (at ", __FILE__, ":", errline, ")"));
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
