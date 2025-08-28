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
#include <sstream>
#include <stdexcept>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "test/cpp/cpp_test_util.h"
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

namespace cpp_test {

// Prefix of the C++ stacktrace PyTorch adds to the error message.
constexpr inline char kTorchCppStacktracePrefixDeprecated[] =
    "Exception raised from OkOrThrow at torch_xla/csrc/status.cpp:";
constexpr inline char kTorchCppStacktracePrefix[] =
    "Exception raised from ThrowStatusError at torch_xla/csrc/status.cpp:";

constexpr inline char kNewMessage[] = "New test error message";
constexpr inline char kMessage[] = "Test error message";
constexpr inline char kFile[] = "test_file.cpp";
constexpr inline char kFunction[] = "foo";
constexpr inline char kEntryPrefix[] = "\n    ";
constexpr inline int32_t kLine = 42;

inline std::string GetStatusPropagationTrace(const absl::Status& status) {
  if (status.ok()) {
    return "";
  }
  auto status_propagation_trace = status.GetPayload(kStatusPropagationTraceKey);
  return status_propagation_trace.has_value()
             ? std::string(status_propagation_trace->Flatten())
             : "";
}

TEST_P(StatusTest, OkOrThrowWithOkStatus) {
  absl::Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(OkOrThrow(ok_status));
}

TEST_P(StatusTest, OkOrThrowWithErrorStatus) {
  try {
    absl::Status error_status = absl::InvalidArgumentError(kMessage);
    OkOrThrow(error_status);
  } catch (const c10::Error& error) {
    if (IsShowCppStacktracesMode()) {
      EXPECT_THAT(std::string_view(error.what()),
                  ::testing::StartsWith(absl::StrCat(
                      kMessage, "\n\n", kTorchCppStacktracePrefixDeprecated)));
    } else {
      EXPECT_EQ(std::string_view(error.what_without_backtrace()),
                std::string_view(kMessage));
    }
  }
}

TEST_P(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  absl::StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST_P(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  try {
    absl::StatusOr<int> error_status = absl::InvalidArgumentError(kMessage);
    int value = GetValueOrThrow(error_status);
  } catch (const c10::Error& error) {
    if (IsShowCppStacktracesMode()) {
      EXPECT_THAT(std::string_view(error.what()),
                  ::testing::StartsWith(absl::StrCat(
                      kMessage, "\n\n", kTorchCppStacktracePrefixDeprecated)));
    } else {
      EXPECT_EQ(std::string_view(error.what_without_backtrace()),
                std::string_view(kMessage));
    }
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
    std::ostringstream oss;
    oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":" << errline0
        << " (error: " << kMessage << ")";
    oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
        << errline1;
    oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
        << errline2;
    EXPECT_EQ(GetStatusPropagationTrace(result), oss.str());
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

TEST_P(StatusTest, OkOrThrowWithErrorPropagationWithNewMessage) {
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

  try {
    OkOrThrow(outerfn());
  } catch (const c10::Error& error) {
    if (IsShowCppStacktracesMode()) {
      // Expected Error Message Prefix
      // =============================
      //
      // New test error kMessage
      //
      // Status Propagation Stacktrace:
      //     From: ./test/cpp/test_status_common.h:329 (error: Test error
      //     kMessage) From: ./test/cpp/test_status_common.h:335 (error: New
      //     test error kMessage) From: ./test/cpp/test_status_common.h:342
      //
      // C++ Stacktrace:
      //
      std::ostringstream oss;
      oss << kNewMessage;
      oss << "\n\n";
      oss << "Status Propagation Trace:";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline0 << " (error: " << kMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline1 << " (error: " << kNewMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline2;
      oss << "\n\n";
      oss << kTorchCppStacktracePrefixDeprecated;
      EXPECT_THAT(std::string_view(error.what()),
                  ::testing::StartsWith(oss.str()));
    } else {
      EXPECT_EQ(std::string_view(error.what_without_backtrace()),
                std::string_view(kNewMessage));
    }
  }
}

TEST_P(StatusTest, MacroThrowIfErrorWithErrorPropagationWithNewMessage) {
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

  int32_t errline3 = __LINE__ + 2;
  try {
    XLA_THROW_IF_ERROR(outerfn());
    FAIL() << "Expected `XLA_THROW_IF_ERROR(outerfn())` to throw.";
  } catch (const c10::Error& error) {
    if (IsShowCppStacktracesMode()) {
      // clang-format off
      //
      // Expected Error Message Prefix
      // =============================
      //
      // New test error kMessage
      //
      // Status Propagation Stacktrace:
      //     From: operator() at ./test/cpp/test_status_common.h:334 (error: Test error kMessage)
      //     From: operator() at ./test/cpp/test_status_common.h:339 (error: New test error kMessage)
      //     From: operator() at ./test/cpp/test_status_common.h:345
      //     From: TestBody at ./test/cpp/test_status_common.h:350
      //
      // C++ Stacktrace:
      //
      // clang-format on
      std::ostringstream oss;
      oss << kNewMessage;
      oss << "\n\n";
      oss << "Status Propagation Trace:";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline0 << " (error: " << kMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline1 << " (error: " << kNewMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline2;
      oss << kEntryPrefix << "From: TestBody at " << __FILE__ << ":"
          << errline3;
      oss << "\n\n";
      oss << kTorchCppStacktracePrefix;
      EXPECT_THAT(std::string_view(error.what()),
                  ::testing::StartsWith(oss.str()));
    } else {
      EXPECT_EQ(std::string_view(error.what_without_backtrace()),
                std::string_view(kNewMessage));
    }
  }
}

TEST_P(StatusTest, MacroAssignOrThrowWithErrorPropagationWithNewMessage) {
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
  auto outerfn = [&]() -> absl::StatusOr<int> {
    XLA_RETURN_IF_ERROR(midfn());
    return 42;
  };

  int32_t errline3 = __LINE__ + 2;
  try {
    XLA_ASSIGN_OR_THROW(int ret, outerfn());
    FAIL() << "Expected `XLA_ASSIGN_OR_THROW(int ret, outerfn())` to throw.";
  } catch (const c10::Error& error) {
    if (IsShowCppStacktracesMode()) {
      // clang-format off
      //
      // Expected Error Message Prefix
      // =============================
      //
      // New test error kMessage
      //
      // Status Propagation Stacktrace:
      //     From: operator() at ./test/cpp/test_status_common.h:393 (error: Test error kMessage)
      //     From: operator() at ./test/cpp/test_status_common.h:398 (error: New test error kMessage)
      //     From: operator() at ./test/cpp/test_status_common.h:404
      //     From: TestBody at ./test/cpp/test_status_common.h:410
      //
      // C++ Stacktrace:
      //
      // clang-format on
      std::ostringstream oss;
      oss << kNewMessage;
      oss << "\n\n";
      oss << "Status Propagation Trace:";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline0 << " (error: " << kMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline1 << " (error: " << kNewMessage << ")";
      oss << kEntryPrefix << "From: operator() at " << __FILE__ << ":"
          << errline2;
      oss << kEntryPrefix << "From: TestBody at " << __FILE__ << ":"
          << errline3;
      oss << "\n\n";
      oss << kTorchCppStacktracePrefix;
      EXPECT_THAT(std::string_view(error.what()),
                  ::testing::StartsWith(oss.str()));
    } else {
      EXPECT_EQ(std::string_view(error.what_without_backtrace()),
                std::string_view(kNewMessage));
    }
  }
}

}  // namespace cpp_test
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_TEST_STATUS_COMMON_H_
