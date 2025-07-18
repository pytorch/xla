// This file centralizes testing for status propagation functions and macros.
//
// It contains classes, functions, and actual tests. These tests are written
// in a parameterized way, so as to avoid duplicating tests for both
// configurations:
//
//   1. `TORCH_SHOW_CPP_STACKTRACES=true`: instantiated in
//      test_status_show_cpp_stacktraces.cpp
//
//   2. `TORCH_SHOW_CPP_STACKTRACES=false`: instantiated in
//      test_status_dont_show_cpp_stacktraces.cpp
//
// In order to easily instantiate the tests, this file also defines the macro
// `INSTANTIATE_TEST_SUITE_WITH_MODE(mode)`, where `mode` is either `kShow` or
// `kHide`, for actually instantiating these tests with the environment variable
// set to the correct value.

#ifndef XLA_TEST_CPP_TEST_STATUS_COMMON_H_
#define XLA_TEST_CPP_TEST_STATUS_COMMON_H_

#include <gtest/gtest.h>

#include <cstdlib>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

// Enum to control whether C++ error context is shown in status messages
enum class CppStacktracesMode {
  kShow,
  kHide,
};

// Converts CppStacktracesMode enum to string for test parameter naming
inline const char* const ToString(CppStacktracesMode mode) {
  switch (mode) {
    case CppStacktracesMode::kShow:
      return "ShowCppStacktraces";
    case CppStacktracesMode::kHide:
      return "DontShowCppStacktraces";
  }
}

// Base test class for parameterized status tests with C++ error context control
class StatusTest : public testing::TestWithParam<CppStacktracesMode> {
 public:
  StatusTest() {
    const char* const value = IsShowCppStacktracesMode() ? "1" : "0";
    setenv("TORCH_SHOW_CPP_STACKTRACES", value, /* replace= */ 1);
  }

 protected:
  bool IsShowCppStacktracesMode() {
    return GetParam() == CppStacktracesMode::kShow;
  }
};

// Macro to instantiate parameterized status tests with specific mode
//
// Note that the `test` parameter of this macro requires that it should be a
// non-qualified identifier. That's because the underlying
// `INSTANTIATE_TEST_SUITE_P` GTest macro will concatenate it with other
// things for creating a unique identifier.
#define INSTANTIATE_WITH_CPP_STACKTRACES_MODE(suite, test, mode)             \
  INSTANTIATE_TEST_SUITE_P(                                                  \
      suite, test, ::testing::Values(::torch_xla::CppStacktracesMode::mode), \
      [](const ::testing::TestParamInfo<::torch_xla::CppStacktracesMode>&    \
             info) { return ToString(info.param); })

namespace testing {

constexpr inline char new_message[] = "New test error message";
constexpr inline char message[] = "Test error message";
constexpr inline char test_file[] = "test_file.cpp";
constexpr inline int32_t line = 42;

TEST_P(StatusTest, MaybeThrowWithOkStatus) {
  absl::Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST_P(StatusTest, MaybeThrowWithErrorStatus) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST_P(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  absl::StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST_P(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  absl::StatusOr<int> status_or = absl::InvalidArgumentError(message);
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST_P(StatusTest, MaybeWithLocationPropagatesErrorStatus) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result = MaybeWithLocation(error_status, test_file, line);
  if (IsShowCppStacktracesMode()) {
    ASSERT_NE(result, error_status);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code(), error_status.code());
    EXPECT_EQ(result.message(), "Test error message (at test_file.cpp:42)");
  } else {
    EXPECT_EQ(result, error_status);
  }
}

TEST_P(StatusTest, MaybeWithNewMessageEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result = MaybeWithNewMessage(error_status, test_file, line);
  EXPECT_EQ(result, error_status);
}

TEST_P(StatusTest, MaybeWithNewMessageNonEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  absl::Status result =
      MaybeWithNewMessage(error_status, test_file, line, new_message);

  ASSERT_NE(result, error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(result.message(),
              absl::StrCat("New test error message (at test_file.cpp:42)\n"
                           "From Error: Test error message"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroReturnIfError) {
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

TEST_P(StatusTest, MacroReturnIfErrorWithError) {
  auto test_function = [=]() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(message);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(message));
}

TEST_P(StatusTest, MacroReturnIfErrorWithNestedError) {
  int32_t errline = 0;
  auto inner_test_function = [&errline]() -> absl::Status {
    errline = __LINE__ + 1;
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(message));
  };

  auto test_function = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(inner_test_function());
    return absl::OkStatus();
  };

  auto outer_test_function = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(test_function());
    return absl::OkStatus();
  };

  absl::Status result = outer_test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(result.message(), absl::StrCat("Test error message (at ",
                                             __FILE__, ":", errline, ")"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(message));
  }
}

TEST_P(StatusTest, MacroReturnIfErrorWithErrorWithNewMessage) {
  int32_t errline = 0;
  auto test_function = [&errline]() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(message);
    errline = __LINE__ + 1;
    XLA_RETURN_IF_ERROR(error_status, new_message);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(result.message(),
              absl::StrCat("New test error message (at ", __FILE__, ":",
                           errline, ")\nFrom Error: Test error message"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroAssignOrReturn) {
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

TEST_P(StatusTest, MacroAssignOrReturnWithError) {
  auto test_function = []() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError(message);
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), std::string_view(message));
}

TEST_P(StatusTest, MacroAssignOrReturnWithErrorWithNewMessage) {
  int32_t errline = 0;

  auto test_function = [&errline]() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError(message);
    errline = __LINE__ + 1;
    XLA_ASSIGN_OR_RETURN(int value, status_or, new_message);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(result.status().message(),
              absl::StrCat("New test error message (at ", __FILE__, ":",
                           errline, ")\nFrom Error: Test error message"));
  } else {
    EXPECT_EQ(result.status().message(), std::string_view(new_message));
  }
}

TEST_P(StatusTest, MacroErrorWithLocation) {
  absl::Status error_status = absl::InvalidArgumentError(message);
  int32_t errline = __LINE__ + 1;
  absl::Status result = XLA_ERROR_WITH_LOCATION(error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(result.message(), absl::StrCat("Test error message (at ",
                                             __FILE__, ":", errline, ")"));
  } else {
    EXPECT_EQ(result.message(), std::string_view(message));
  }
}

}  // namespace testing
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_TEST_STATUS_COMMON_H_
