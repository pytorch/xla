#ifndef XLA_TEST_CPP_TEST_STATUS_COMMON_H_
#define XLA_TEST_CPP_TEST_STATUS_COMMON_H_

#include <gtest/gtest.h>

#include <cstdlib>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

// Enum to control whether C++ error context is shown in status messages
enum class CppErrorContextMode {
  SHOW,
  HIDE,
};

// Converts CppErrorContextMode enum to string for test parameter naming
inline const char* const ToString(CppErrorContextMode mode) {
  switch (mode) {
    case CppErrorContextMode::SHOW:
      return "ShowCppErrorContext";
    case CppErrorContextMode::HIDE:
      return "DontShowCppErrorContext";
  }
}

// Base test class for parameterized status tests with C++ error context control
class StatusTest : public testing::TestWithParam<CppErrorContextMode> {
  void SetUp() override {
    const char* const value = IsShowCppErrorContextMode() ? "true" : "false";
    setenv(runtime::env::kEnvShowCppErrorContext, value, /* replace= */ 1);
  }

 public:
  bool IsShowCppErrorContextMode() {
    return GetParam() == CppErrorContextMode::SHOW;
  }
};

// Macro to instantiate parameterized status tests with specific mode
#define INSTANTIATE_TEST_SUITE_WITH_MODE(mode)                            \
  using torch_xla::CppErrorContextMode;                                   \
  using torch_xla::StatusTest;                                            \
  INSTANTIATE_TEST_SUITE_P(                                               \
      StatusTest, StatusTest, testing::Values(CppErrorContextMode::mode), \
      [](const testing::TestParamInfo<CppErrorContextMode>& info) {       \
        return ToString(info.param);                                      \
      })

namespace {

using absl::Status;
using absl::StatusCode;
using absl::StatusOr;
using absl::StrCat;

constexpr char new_message[] = "New test error message";
constexpr char message[] = "Test error message";
constexpr char test_file[] = "test_file.cpp";
constexpr int32_t line = 42;

TEST_P(StatusTest, MaybeThrowWithOkStatus) {
  Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST_P(StatusTest, MaybeThrowWithErrorStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST_P(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST_P(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  StatusOr<int> status_or = absl::InvalidArgumentError(message);
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST_P(StatusTest, MaybeWithLocationPropagatesErrorStatus) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithLocation(error_status, test_file, line);
  if (IsShowCppErrorContextMode()) {
    ASSERT_NE(result, error_status);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code(), error_status.code());
    EXPECT_EQ(result.message(), "Test error message (at test_file.cpp:42)");
  } else {
    EXPECT_EQ(result, error_status);
  }
}

TEST_P(StatusTest, MaybeWithNewMessageEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result = MaybeWithNewMessage(error_status, test_file, line);
  EXPECT_EQ(result, error_status);
}

TEST_P(StatusTest, MaybeWithNewMessageNonEmptyNewMessage) {
  Status error_status = absl::InvalidArgumentError(message);
  Status result =
      MaybeWithNewMessage(error_status, test_file, line, new_message);

  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());

  if (IsShowCppErrorContextMode()) {
    EXPECT_EQ(result.message(),
              StrCat("New test error message (at test_file.cpp:42)\n"
                     "From Error: Test error message"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroReturnIfError) {
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

TEST_P(StatusTest, MacroReturnIfErrorWithError) {
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

TEST_P(StatusTest, MacroReturnIfErrorWithNestedError) {
  int32_t errline = 0;
  auto inner_test_function = [&errline]() -> Status {
    errline = __LINE__ + 1;
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(message));
  };

  auto test_function = [&]() -> Status {
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

  if (IsShowCppErrorContextMode()) {
    EXPECT_EQ(result.message(),
              StrCat("Test error message (at ", __FILE__, ":", errline, ")"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(message));
  }
}

TEST_P(StatusTest, MacroReturnIfErrorWithErrorWithNewMessage) {
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

  if (IsShowCppErrorContextMode()) {
    EXPECT_EQ(result.message(),
              StrCat("New test error message (at ", __FILE__, ":", errline,
                     ")\nFrom Error: Test error message"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroAssignOrReturn) {
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

TEST_P(StatusTest, MacroAssignOrReturnWithError) {
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

TEST_P(StatusTest, MacroAssignOrReturnWithErrorWithNewMessage) {
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

  if (IsShowCppErrorContextMode()) {
    EXPECT_EQ(result.status().message(),
              StrCat("New test error message (at ", __FILE__, ":", errline,
                     ")\nFrom Error: Test error message"));
  } else {
    EXPECT_EQ(result.status().message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroErrorWithLocation) {
  Status error_status = absl::InvalidArgumentError(message);
  int32_t errline = __LINE__ + 1;
  Status result = XLA_ERROR_WITH_LOCATION(error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
  if (IsShowCppErrorContextMode()) {
    EXPECT_EQ(result.message(),
              StrCat("Test error message (at ", __FILE__, ":", errline, ")"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(message));
  }
}

}  // namespace
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_TEST_STATUS_COMMON_H_
