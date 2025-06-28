#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

TEST(StatusTest, MaybeWithLocationRetunsSameStatus) {
  absl::Status error_status = absl::InvalidArgumentError("Test error message");
  absl::Status result = MaybeWithLocation(error_status, "test_file.cpp", 42);
  EXPECT_EQ(result, error_status);
}

TEST(StatusTest, MaybeWithNewMessageEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError("Original error");
  absl::Status result = MaybeWithNewMessage(error_status, "test_file.cpp", 42);
  EXPECT_EQ(result, error_status);
}

TEST(StatusTest, MaybeWithNewMessageNonEmptyNewMessage) {
  constexpr char new_err_string[] = "New error message";
  absl::Status error_status = absl::InvalidArgumentError("Original error");
  absl::Status result =
      MaybeWithNewMessage(error_status, "test_file.cpp", 42, new_err_string);

  ASSERT_FALSE(result.ok());
  ASSERT_NE(result, error_status);
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), new_err_string);
}

TEST(StatusTest, MaybeThrowWithOkStatus) {
  absl::Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST(StatusTest, MaybeThrowWithErrorStatus) {
  absl::Status error_status = absl::InvalidArgumentError("Test error");
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  absl::StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  absl::StatusOr<int> status_or = absl::InvalidArgumentError("Test error");
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST(StatusTest, MacroReturnIfError) {
  int value = 42;

  auto test_function = [=]() -> absl::StatusOr<int> {
    absl::Status ok_status = absl::OkStatus();
    XLA_RETURN_IF_ERROR(ok_status);
    return value;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), value);
}

TEST(StatusTest, MacroReturnIfErrorWithError) {
  constexpr char err_string[] = "Test error";

  auto test_function = [=]() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(err_string);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), err_string);
}

TEST(StatusTest, MacroAssignOrReturn) {
  int initial_value = 42;
  int expected_value = initial_value * 2;

  auto test_function = [=]() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = initial_value;
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), expected_value);
}

TEST(StatusTest, MacroAssignOrReturnWithError) {
  auto test_function = []() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError("Test error");
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(StatusTest, MacroErrorWithLocation) {
  absl::Status error_status = absl::InvalidArgumentError("Test error");
  absl::Status result = XLA_ERROR_WITH_LOCATION(error_status);
  EXPECT_EQ(result, error_status);
}

}  // namespace torch_xla
