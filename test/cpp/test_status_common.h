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

#include <c10/util/Exception.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

// Enum to control whether C++ error context is shown in status messages.
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

constexpr inline char kNewMessage[] = "New test error message";
constexpr inline char kMessage[] = "Test error message";
constexpr inline char kFile[] = "test_file.cpp";
constexpr inline char kFunction[] = "foo";
constexpr inline char kEntryPrefix[] = "\n    ";
constexpr inline int32_t kLine = 42;

// The PyTorch C++ stacktrace is ALWAYS appended to the error message.
// More specifically, when `what()` function is called.
//
// However, it's only when the raised `c10::Error` gets translated to a
// Python exception that PyTorch checks the value of the
// `TORCH_SHOW_CPP_STACKTRACES` environment variable, which actually
// controls whether the stacktrace will get shown or not by calling
// `what_without_backtraces()`, instead.
//
// Therefore, we need to mimic this behavior.
#define THROW_RUNTIME_ERROR_FROM_C10_ERROR(block)                   \
  try {                                                             \
    block;                                                          \
  } catch (const c10::Error& error) {                               \
    throw std::runtime_error(IsShowCppStacktracesMode()             \
                                 ? error.what()                     \
                                 : error.what_without_backtrace()); \
  }

// Prefix of the C++ stacktrace PyTorch adds to the error message.
constexpr inline char kTorchCppStacktracePrefix[] =
    "Exception raised from MaybeThrow at torch_xla/csrc/status.cpp:";

inline std::string GetStatusPropagationTrace(const absl::Status& status) {
  if (status.ok()) {
    return "";
  }
  auto status_propagation_trace = status.GetPayload(kStatusPropagationTraceKey);
  return status_propagation_trace.has_value()
             ? std::string(status_propagation_trace->Flatten())
             : "";
}

TEST_P(StatusTest, MaybeThrowWithOkStatus) {
  absl::Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST_P(StatusTest, MaybeThrowWithErrorStatus) {
  auto throw_exception = [=]() {
    THROW_RUNTIME_ERROR_FROM_C10_ERROR({
      absl::Status error_status = absl::InvalidArgumentError(kMessage);
      MaybeThrow(error_status);
    });
  };

  if (IsShowCppStacktracesMode()) {
    std::string expected_prefix =
        absl::StrCat(kMessage, "\n\n", kTorchCppStacktracePrefix);
    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::StartsWith(expected_prefix)));
  } else {
    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::Eq(kMessage)));
  }
}

TEST_P(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  absl::StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST_P(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  auto throw_exception = [=]() {
    THROW_RUNTIME_ERROR_FROM_C10_ERROR({
      absl::StatusOr<int> error_status = absl::InvalidArgumentError(kMessage);
      int value = GetValueOrThrow(error_status);
    });
  };
  if (IsShowCppStacktracesMode()) {
    std::string expected_prefix =
        absl::StrCat(kMessage, "\n\n", kTorchCppStacktracePrefix);
    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::StartsWith(expected_prefix)));
  } else {
    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::Eq(kMessage)));
  }
}

TEST_P(StatusTest, MaybeWithLocationPropagatesErrorStatus) {
  absl::Status error_status = absl::InvalidArgumentError(kMessage);
  absl::Status result =
      status_internal::MaybeWithLocation(error_status, kFile, kLine, kFunction);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), error_status.message());

  if (IsShowCppStacktracesMode()) {
    EXPECT_NE(result, error_status);
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(kEntryPrefix, "From: ", kFunction, " at ", kFile,
                           ":", kLine, " (error: ", kMessage, ")"));
  } else {
    EXPECT_EQ(result, error_status);
  }
}

TEST_P(StatusTest, MaybeWithNewMessageEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(kMessage);
  absl::Status result = status_internal::MaybeWithNewMessage(
      error_status, kFile, kLine, kFunction);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), error_status.message());

  if (IsShowCppStacktracesMode()) {
    EXPECT_NE(result, error_status);
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(kEntryPrefix, "From: ", kFunction, " at ", kFile,
                           ":", kLine));
  } else {
    EXPECT_EQ(result, error_status);
  }
}

TEST_P(StatusTest, MaybeWithNewMessageNonEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError(kMessage);
  absl::Status result = status_internal::MaybeWithNewMessage(
      error_status, kFile, kLine, kFunction, kNewMessage);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), std::string_view(kNewMessage));
  EXPECT_NE(result, error_status);

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(kEntryPrefix, "From: ", kFunction, " at ", kFile,
                           ":", kLine, " (error: ", kNewMessage, ")"));
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
    absl::Status error_status = absl::InvalidArgumentError(kMessage);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(kMessage));
}

TEST_P(StatusTest, MacroReturnIfErrorWithNestedError) {
  int32_t errline0 = __LINE__ + 2;
  auto inner_test_function = []() -> absl::Status {
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(kMessage));
  };

  int32_t errline1 = __LINE__ + 2;
  auto test_function = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(inner_test_function());
    return absl::OkStatus();
  };

  int32_t errline2 = __LINE__ + 2;
  auto outer_test_function = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(test_function());
    return absl::OkStatus();
  };

  absl::Status result = outer_test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(kMessage));

  if (IsShowCppStacktracesMode()) {
    auto frame0 = absl::StrCat(kEntryPrefix, "From: operator() at ", __FILE__,
                               ":", errline0, " (error: ", kMessage, ")");
    auto frame1 = absl::StrCat(kEntryPrefix, "From: operator() at ", __FILE__,
                               ":", errline1);
    auto frame2 = absl::StrCat(kEntryPrefix, "From: operator() at ", __FILE__,
                               ":", errline2);
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(frame0, frame1, frame2));
  }
}

TEST_P(StatusTest, MacroReturnIfErrorWithErrorWithNewMessage) {
  int32_t errline = __LINE__ + 3;
  auto test_function = []() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(kMessage);
    XLA_RETURN_IF_ERROR(error_status, kNewMessage);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(kNewMessage));

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(kEntryPrefix, "From: operator() at ", __FILE__, ":",
                           errline, " (error: ", kNewMessage, ")"));
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
    absl::StatusOr<int> status_or = absl::InvalidArgumentError(kMessage);
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), std::string_view(kMessage));
}

TEST_P(StatusTest, MacroAssignOrReturnWithErrorWithNewMessage) {
  int32_t errline = __LINE__ + 3;
  auto test_function = []() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError(kMessage);
    XLA_ASSIGN_OR_RETURN(int value, status_or, kNewMessage);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), std::string_view(kNewMessage));

  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(GetStatusPropagationTrace(result.status()),
              absl::StrCat(kEntryPrefix, "From: operator() at ", __FILE__, ":",
                           errline, " (error: ", kNewMessage, ")"));
  }
}

TEST_P(StatusTest, MacroErrorWithLocation) {
  absl::Status error_status = absl::InvalidArgumentError(kMessage);
  int32_t errline = __LINE__ + 1;
  absl::Status result = XLA_ERROR_WITH_LOCATION(error_status);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), std::string_view(kMessage));
  if (IsShowCppStacktracesMode()) {
    EXPECT_EQ(GetStatusPropagationTrace(result),
              absl::StrCat(kEntryPrefix, "From: ", __FUNCTION__, " at ",
                           __FILE__, ":", errline, " (error: ", kMessage, ")"));
  }
}

TEST_P(StatusTest, MaybeThrowWithErrorPropagationWithNewMessage) {
  int32_t errline0 = __LINE__ + 2;
  auto innerfn = [&]() -> absl::Status {
    return XLA_ERROR_WITH_LOCATION(absl::InvalidArgumentError(kMessage));
  };

  int32_t errline1 = __LINE__ + 2;
  auto midfn = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(innerfn(), kNewMessage);
    return absl::OkStatus();
  };

  int32_t errline2 = __LINE__ + 2;
  auto outerfn = [&]() -> absl::Status {
    XLA_RETURN_IF_ERROR(midfn());
    return absl::OkStatus();
  };

  auto throw_exception = [&]() {
    THROW_RUNTIME_ERROR_FROM_C10_ERROR(MaybeThrow(outerfn()));
  };

  if (IsShowCppStacktracesMode()) {
    // Expected Error Message Prefix
    // =============================
    //
    // New test error kMessage
    //
    // Status Propagation Stacktrace:
    //     From: ./test/cpp/test_status_common.h:329 (error: Test error
    //     kMessage) From: ./test/cpp/test_status_common.h:335 (error: New test
    //     error kMessage) From: ./test/cpp/test_status_common.h:342
    //
    // C++ Stacktrace:
    //
    std::string expected_prefix = absl::StrCat(
        kNewMessage, "\n\nStatus Propagation Trace:", kEntryPrefix,
        "From: operator() at ", __FILE__, ":", errline0, " (error: ", kMessage,
        ")", kEntryPrefix, "From: operator() at ", __FILE__, ":", errline1,
        " (error: ", kNewMessage, ")", kEntryPrefix, "From: operator() at ",
        __FILE__, ":", errline2, "\n\n", kTorchCppStacktracePrefix);

    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::StartsWith(expected_prefix)));
  } else {
    EXPECT_THAT(throw_exception, ::testing::ThrowsMessage<std::runtime_error>(
                                     ::testing::Eq(kNewMessage)));
  }
}

}  // namespace testing
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_TEST_STATUS_COMMON_H_
