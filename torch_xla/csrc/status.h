// This file provides utilities for handling error statuses in the PyTorch/XLA
// project.
//
// The main features are:
// - Macros for creating error statuses with location information (__FILE__ and
// __LINE__)
// - Functions for propagating errors while optionally capturing new messages
// - Utilities for modifying existing status messages with additional context

#ifndef XLA_TORCH_XLA_CSRC_STATUS_H_
#define XLA_TORCH_XLA_CSRC_STATUS_H_

#include <sstream>

#include "absl/status/statusor.h"

namespace torch_xla {

// `type_url` for retrieving the status propagation trace payload of a given
// status.
//
// The payload is composed of multiple lines, where each line represents a stack
// frame in the status propagation trace. Each line is in the following format:
//
//     \n    From: <file>:<line>[ErrorSuffix]
//     | ----                     |
//     |  |                       |_ error message produced in that source
//     |  |                          location (it might be overwritten later).
//     |  |
//     |  |_ leading 4 spaces for improved readability.
//     |
//     |_ start with a line break.
//
constexpr char kStatusPropagationTraceKey[] =
    "type.googleapis.com/torch_xla.status_trace";

// If `TORCH_SHOW_CPP_STACKTRACES` is set, creates a new Status instance,
// appending the current location (e.g. file and line information) to the
// status message.
//
// This should be used whenever we are returning new error status.
//
// Example:
//
//     XLA_ERROR_WITH_LOCATION(
//         absl::InvalidArgumentError("Error message.")
//     );
//
// If `TORCH_SHOW_CPP_STACKTRACES` is set, the error shown will be:
//
//     RuntimeError: Error message.
//       From: <cpp-source-file>:<line> (error: Error message.)
//
#define XLA_ERROR_WITH_LOCATION(status)                                       \
  ::torch_xla::status_internal::MaybeWithLocation(status, __FILE__, __LINE__, \
                                                  __FUNCTION__)

#define XLA_CONCAT_(a, b) XLA_CONCAT_IMPL_(a, b)
#define XLA_CONCAT_IMPL_(a, b) a##b

// Unique identifier for the status variable for the current line.
#define XLA_STATUS_VAR_ XLA_CONCAT_(status_, __LINE__)

// Provides a flexible way to handle error checking with optional message
// modification. It evaluates `expr`, and:
//
//   1. Runs the `on_error` block, if the returned status is an error
//   2. Runs the `on_success` block, otherwise
//
#define XLA_PROCESS_STATUS_IMPL_(on_error, on_success, expr, var, ...) \
  auto var = (expr);                                                   \
  if (!var.ok()) {                                                     \
    on_error(var, ##__VA_ARGS__);                                      \
  }                                                                    \
  on_success

// `on_error` implementation for propagating the status `var`.
//
// This macro wraps `var` (error status returned) into a new status, adding
// source location information to the status propagation trace if
// `TORCH_SHOW_CPP_STACKTRACES` is set. And then, returns the newly created
// status.
//
// It should be only used as parameter to `XLA_PROCESS_STATUS_IMPL_` macro
// defined above.
//
#define XLA_PROPAGATE_STATUS_IMPL_(var, ...)                            \
  return ::torch_xla::status_internal::MaybeWithNewMessage(             \
      ::torch_xla::status_internal::GetStatus(var), __FILE__, __LINE__, \
      __FUNCTION__, ##__VA_ARGS__)

// `on_error` implementation for throwing an exception with the status `var`.
//
// This macro wraps `var` (error status returned) into a new status, adding
// source location information to the status propagation trace if
// `TORCH_SHOW_CPP_STACKTRACES` is set. And then, throws an exception using the
// `ThrowStatusError()` function.
//
// It should be only used as parameter to `XLA_PROCESS_STATUS_IMPL_` macro
// defined above.
//
#define XLA_THROW_STATUS_IMPL_(var, ...)                                \
  ::torch_xla::status_internal::ThrowStatusError(                       \
      ::torch_xla::status_internal::GetStatus(var), __FILE__, __LINE__, \
      __FUNCTION__, ##__VA_ARGS__)

// Macro implementation for processing an `absl::Status` value. This is the core
// definition of `XLA_*_IF_ERROR()` macros that, given that `rexpr` is an error
// status, either throws or returns (i.e. propagates) a newly created status
// with source location information.
//
// If `rexpr` results in an ok status, execution continues. Otherwise, we run
// `on_error`. Then, if `TORCH_SHOW_CPP_STACKTRACES` is set, the error shown
// will be:
//
//     RuntimeError: New error message.
//
//     Status Propagation Stacktrace:
//       ...
//       From: <cpp-source-file>:<line> (error: Previous error message.)
//       ...
//       From: <cpp-source-file>:<line> (error: New error message.)
//
#define XLA_DO_IF_ERROR_IMPL_(on_error, rexpr, ...)                 \
  do {                                                              \
    XLA_PROCESS_STATUS_IMPL_(on_error, /* on_success= */ {}, rexpr, \
                             XLA_STATUS_VAR_, ##__VA_ARGS__)        \
  } while (false)

// If `rexpr` returns a non-ok status, this macro propagates the returned status
// by early-returning a, possibly, new status with source location information.
// Otherwise, continues execution.
//
// Example:
//
//     XLA_RETURN_IF_ERROR(
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
#define XLA_RETURN_IF_ERROR(rexpr, ...) \
  XLA_DO_IF_ERROR_IMPL_(XLA_PROPAGATE_STATUS_IMPL_, rexpr, ##__VA_ARGS__)

// If `rexpr` returns a non-ok status, this macro throws an exception with the
// returned status, possibly, wrapped by a new status with source location
// information. Otherwise, continues execution.
//
// Example:
//
//     XLA_THROW_IF_ERROR(
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
#define XLA_THROW_IF_ERROR(rexpr, ...) \
  XLA_DO_IF_ERROR_IMPL_(XLA_THROW_STATUS_IMPL_, rexpr, ##__VA_ARGS__)

// Macro implementation for processing an `absl::Status` value. This is the core
// definition of `XLA_ASSIGN_OR_*()` macros that, given that `rexpr` is an error
// status, either throws or returns (i.e. propagates) a newly created status
// with source location information.
//
// If `rexpr` results in an ok status, we assign the value held by the status
// returned by `rexpr` to `lhs`. Otherwise, we run `on_error`.
//
// Note 1: `lhs` might be a variable declarate, e.g:
//
// Note 2: this macro will be replaced by multiple statements that live on
//         the scope it was called (see `XLA_PROCESS_STATUS_IMPL_`).
//
#define XLA_ASSIGN_OR_DO_IMPL_(on_error, lhs, rexpr, ...)                   \
  XLA_PROCESS_STATUS_IMPL_(                                                 \
      on_error, /* on_success= */ lhs = std::move(XLA_STATUS_VAR_).value(), \
      rexpr, XLA_STATUS_VAR_, ##__VA_ARGS__)

// If `rexpr` returns a non-ok status, this macro propagates the returned status
// by early-returning a, possibly, new status with source location information.
// Otherwise, assigns `rexpr` to `lhs`.
//
// Example:
//
//     XLA_ASSIGN_OR_RETURN(
//         int result,
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
#define XLA_ASSIGN_OR_RETURN(lhs, rexpr, ...) \
  XLA_ASSIGN_OR_DO_IMPL_(XLA_PROPAGATE_STATUS_IMPL_, lhs, rexpr, ##__VA_ARGS__)

// If `rexpr` returns a non-ok status, this macro throws an exception with the
// returned status, possibly, wrapped by a new status with source location
// information. Otherwise, assigns `rexpr` to `lhs`.
//
// Example:
//
//     XLA_ASSIGN_OR_THROW(
//         int result,
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
#define XLA_ASSIGN_OR_THROW(lhs, rexpr, ...) \
  XLA_ASSIGN_OR_DO_IMPL_(XLA_THROW_STATUS_IMPL_, lhs, rexpr, ##__VA_ARGS__)

// Crashes if `status` is not an ok status.
//
// Example:
//
//     XLA_CHECK_OK(
//         FnThatReturnStatus(),
//         "New error message"
//     );
//
// If `FnThatReturnStatus()` returns a non-ok status, this macro will
// call `ABSL_CHECK()`, which will crash.
//
#define XLA_CHECK_OK(status, ...)                                          \
  ::torch_xla::status_internal::OkOrDie(                                   \
      ::torch_xla::status_internal::GetStatus(status), __FILE__, __LINE__, \
      __FUNCTION__, ##__VA_ARGS__)

namespace status_internal {

// Adds source location information to the status propagation trace if
// `TORCH_SHOW_CPP_STACKTRACES` is set.
//
// This function assumes that:
//
//   1. `status` is a non-ok status.
//   2. `status` doesn't have a status propagation trace payload
//
// If any of the above assumptions is false, this function crashes the
// whole program.
//
absl::Status MaybeWithLocation(const absl::Status& status, const char* file,
                               int32_t line, const char* function);

// Returns an `absl::Status` from an `absl::Status`.
// In this case, this function is a no-op. It simply returns the argument.
inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}

// Returns an `absl::Status` from an `absl::StatusOr<T>`.
template <class T>
const absl::Status& GetStatus(const absl::StatusOr<T>& status) {
  return status.status();
}

// Maybe replace the current `status` message with `new_message`, and also
// add source location information if enabled.
//
// This function assumes that `status` is a non-ok status.
//
// The `status` message will get replaced if `new_message` is provided.
// Otherwise, i.e. if it's empty, the old message will remain as the main error
// message.
//
// Rationale: if given, `new_message` has more context, which makes it possible
// to construct better error messages to the user.
//
// This function also appends the source location information to the status
// propagation trace payload (creates a new one if needed), if
// `TORCH_SHOW_CPP_STACKTRACES` is set.
absl::Status MaybeWithNewMessage(const absl::Status& status, const char* file,
                                 int32_t line, const char* function,
                                 std::string_view new_message = "");

// Throws an exception from the given `status`
//
// This function wraps `status` within a new status, with the current source
// location information added to its status propagation trace payload.
//
// Then, it throws an exception by using the `TORCH_CHECK(false)` macro, which
// also displays the C++ stacktrace at the end, if `TORCH_SHOW_CPP_STACKTRACES`
// is set.
void ThrowStatusError(const absl::Status& status, const char* file,
                      const int32_t line, const char* function,
                      std::string_view message = "");

// Checks that `status` is an ok status.
//
// Otherwise, it will create a new status instance with the given source
// location information, and incorporate its message (alongside the
// status propagation trace) to the crash report.
void OkOrDie(const absl::Status& status, const char* file, const int32_t line,
             const char* function, std::string_view message = "");

}  // namespace status_internal

// Builds the complete error message for the given `status`.
//
// If `TORCH_SHOW_CPP_STACKTRACES` is enabled, returns the concatenation of
// `status.message()` with its inner status propagation trace.
//
// It doesn't add a trailing line break.
std::string BuildStatusErrorMessage(const absl::Status& status);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_STATUS_H_
