#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

// Reminder
// ========
//
// This file is a companion to test_status_show_cpp_error_context.cpp.
// This file specifically tests behavior when XLA_SHOW_CPP_ERROR_CONTEXT is
// set to "false".
//
// If you add or delete a test in this file, please make the corresponding
// change in test_status_show_cpp_error_context.cpp as well, adapting for
// XLA_SHOW_CPP_ERROR_CONTEXT being "true" in that file.

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

TEST(StatusWithoutErrorContextTest, MaybeThrowWithOkStatus) {
  Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST(StatusWithoutErrorContextTest, MaybeThrowWithErrorStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST(StatusWithoutErrorContextTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST(StatusWithoutErrorContextTest, GetValueOrThrowWithErrorStatusOr) {
  StatusOr<int> status_or = absl::InvalidArgumentError(message);
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST(StatusWithoutErrorContextTest, MaybeWithLocationRetunsSameStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithLocation(error_status, test_file, line);
  EXPECT_EQ(result, error_status);
}

TEST(StatusWithoutErrorContextTest, MaybeWithNewMessageEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithNewMessage(error_status, test_file, line);
  EXPECT_EQ(result, error_status);
}

TEST(StatusWithoutErrorContextTest, MaybeWithNewMessageNonEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result =
      MaybeWithNewMessage(error_status, test_file, line, new_message);

  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), std::string_view(new_message));
}

TEST(StatusWithoutErrorContextTest, MacroReturnIfError) {
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

TEST(StatusWithoutErrorContextTest, MacroReturnIfErrorWithError) {
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
  auto inner_test_function = []() -> Status {
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(message));
  };

  auto test_function = []() -> Status {
    XLA_RETURN_IF_ERROR(inner_test_function());
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

TEST(StatusWithoutErrorContextTest, MacroReturnIfErrorWithErrorWithNewMessage) {
  auto test_function = [=]() -> Status {
    Status error_status = absl::InvalidArgumentError(message);
    XLA_RETURN_IF_ERROR(error_status, new_message);
    return absl::OkStatus();
  };

  Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(new_message));
}

TEST(StatusWithoutErrorContextTest, MacroAssignOrReturn) {
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

TEST(StatusWithoutErrorContextTest, MacroAssignOrReturnWithError) {
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

TEST(StatusWithoutErrorContextTest,
     MacroAssignOrReturnWithErrorWithNewMessage) {
  auto test_function = []() -> StatusOr<int> {
    StatusOr<int> status_or = absl::InvalidArgumentError(message);
    XLA_ASSIGN_OR_RETURN(int value, status_or, new_message);
    return value * 2;
  };

  StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), std::string_view(new_message));
}

TEST(StatusWithoutErrorContextTest, MacroErrorWithLocation) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = XLA_ERROR_WITH_LOCATION(error_status);
  EXPECT_EQ(result, error_status);
}

void SetUp() {
  setenv(runtime::env::kEnvShowCppErrorContext, /* value= */ "false",
         /* replace= */ 1);
}

}  // namespace
}  // namespace torch_xla

int main(int argc, char **argv) {
  ::torch_xla::SetUp();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
