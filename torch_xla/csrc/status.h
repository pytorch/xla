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

#include "absl/status/statusor.h"

namespace torch_xla {

// Returns whether we should show C++ error context.
//
// More specifically, whether the `TORCH_SHOW_CPP_STACKTRACES` environment
// variable is set or not.
[[nodiscard]] bool ShouldShowCppErrorContext();

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
//     Error message. (at <cpp-source-file>:<line>)
//
#define XLA_ERROR_WITH_LOCATION(status) \
  ::torch_xla::MaybeWithLocation(status, __FILE__, __LINE__)

#define XLA_CONCAT_(a, b) XLA_CONCAT_IMPL_(a, b)
#define XLA_CONCAT_IMPL_(a, b) a##b

// Unique identifier for the status variable for the current line.
#define XLA_STATUS_VAR_ XLA_CONCAT_(status_, __LINE__)

// Provides a flexible way to handle error checking with optional message
// modification. It evaluates `expr`, checks if it's OK, and either:
// 1. Returns early with an error status (potentially modified by the provided
//    additional messages)
// 2. Proceeds with the given `then` block if successful
#define XLA_RETURN_IF_ERROR_IMPL_(expr, var, then, ...)                  \
  auto var = (expr);                                                     \
  if (!var.ok()) {                                                       \
    return ::torch_xla::MaybeWithNewMessage(                             \
        ::torch_xla::GetStatus(var), __FILE__, __LINE__, ##__VA_ARGS__); \
  }                                                                      \
  then

// Propagates `rexpr`, in case it's a non-ok status.
//
// Example:
//
//     XLA_RETURN_IF_ERROR(
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
// If the function call results in an ok status, execution continues. Otherwise,
// we early return a non-ok status. Then, if `TORCH_SHOW_CPP_STACKTRACES` is
// set, the error shown will be:
//
//     New error message. (at <cpp-source-file>:<line>)
//     Previous error message. (at <cpp-source-file>:<line>)
//     ...
//
#define XLA_RETURN_IF_ERROR(rexpr, ...)                                  \
  do {                                                                   \
    XLA_RETURN_IF_ERROR_IMPL_(rexpr, XLA_STATUS_VAR_, {}, ##__VA_ARGS__) \
  } while (false)

// Propagates `rexpr`, in case it's a non-ok status. Otherwise, assign
// its result to `lhs`.
//
// Note 1: `lhs` might be a variable declarate, e.g:
//
// Note 2: this macro will be replaced by multiple statements that live on
//         the scope it was called (see XLA_RETURN_IF_ERROR_IMPL).
//
// Example:
//
//     XLA_ASSIGN_OR_RETURN(
//         int result,
//         FnThatReturnsStatus(),
//         "New error message."
//     );
//
// If the function call results in an ok status, execution continues with
// `result` set to `ret.value()`, where `ret` is the returned value of the
// function. Otherwise, we early return a non-ok status. Then, if
// `TORCH_SHOW_CPP_STACKTRACES` is set, the error shown will be:
//
//     New error message. (at <cpp-source-file>:<line>)
//     Previous error message. (at <cpp-source-file>:<line>)
//     ...
//
#define XLA_ASSIGN_OR_RETURN(lhs, rexpr, ...)                         \
  XLA_RETURN_IF_ERROR_IMPL_(rexpr, XLA_STATUS_VAR_,                   \
                            lhs = std::move(XLA_STATUS_VAR_).value(), \
                            ##__VA_ARGS__)

// Maybe shows location information in the status message.
//
// This function assumes that `status` is a non-ok status.
//
// If `TORCH_SHOW_CPP_STACKTRACES` is set, appends the current source
// location information to the status message. Otherwise, it simply returns
// `status`.
absl::Status MaybeWithLocation(const absl::Status& status, const char* file,
                               int32_t line);

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

// Maybe replace the current `status` message with `new_message`.
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
// This function also appends file location information to the error message, if
// `TORCH_SHOW_CPP_STACKTRACES` is set.
absl::Status MaybeWithNewMessage(const absl::Status& status, const char* file,
                                 int32_t line,
                                 std::string_view new_message = "");

// Maybe throws an exception if `status` has a non-ok code.
//
// Ideally, this function should be used only used in the project's
// boundary, e.g. when we need to throw an exception for the user to see.
void MaybeThrow(const absl::Status& status);

// Either returns the value `status` holds, if it's an ok-status, or throw an
// exception from its error status.
template <class T>
T& GetValueOrThrow(absl::StatusOr<T>& status) {
  MaybeThrow(status.status());
  return status.value();
}

template <class T>
const T& GetValueOrThrow(const absl::StatusOr<T>& status) {
  MaybeThrow(status.status());
  return status.value();
}

template <class T>
T GetValueOrThrow(absl::StatusOr<T>&& status) {
  MaybeThrow(status.status());
  return std::move(status).value();
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_STATUS_H_
