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

// Creates a new Status instance, appending the current location (e.g. file and
// line information) to the status message.
//
// This should be used whenever we are returning new error status, instead of
// propagating. Then, if `XLA_SHOW_CPP_ERROR_CONTEXT` environment variable is
// set, the location information will be shown.
#define XLA_ERROR_WITH_LOCATION(status) \
  ::torch_xla::MaybeWithLocation(status, __FILE__, __LINE__)

#define XLA_CONCAT(a, b) XLA_CONCAT_IMPL(a, b)
#define XLA_CONCAT_IMPL(a, b) a##b

// Unique identifier for the status variable for the current line.
#define XLA_STATUS_VAR XLA_CONCAT(status__, __LINE__)

// Provides a flexible way to handle error checking with optional message
// modification. It evaluates `expr`, checks if it's OK, and either:
// 1. Returns early with an error status (potentially modified by the provided
//    additional messages)
// 2. Proceeds with the given `then` block if successful
#define XLA_RETURN_IF_ERROR_IMPL(expr, var, then, ...)                   \
  auto var = (expr);                                                     \
  if (!var.ok()) {                                                       \
    return ::torch_xla::MaybeWithNewMessage(                             \
        ::torch_xla::GetStatus(var), __FILE__, __LINE__, ##__VA_ARGS__); \
  }                                                                      \
  then;

// Propagates `rexpr`, in case it's a non-ok status.
#define XLA_RETURN_IF_ERROR(rexpr, ...)                                \
  do {                                                                 \
    XLA_RETURN_IF_ERROR_IMPL(rexpr, XLA_STATUS_VAR, {}, ##__VA_ARGS__) \
  } while (false)

// Propagates `rexpr`, in case it's a non-ok status. Otherwise, assign
// its result to `lhs`.
//
// Note 1: `lhs` might be a variable declarate, e.g:
//
//     XLA_ASSIGN_OR_RETURN(int value, FnThatReturnsStatus(), ...);
//
// Note 2: this macro will be replaced by multiple statements that live on
//         the scope it was called (see XLA_RETURN_IF_ERROR_IMPL).
//
#define XLA_ASSIGN_OR_RETURN(lhs, rexpr, ...)                       \
  XLA_RETURN_IF_ERROR_IMPL(rexpr, XLA_STATUS_VAR,                   \
                           lhs = std::move(XLA_STATUS_VAR).value(); \
                           , ##__VA_ARGS__)

// Maybe shows location information in the status message.
//
// This function assumes that `status` is a non-ok status.
//
// If `XLA_SHOW_CPP_ERROR_CONTEXT` is set, appends the current source
// location information to the status message. Otherwise, it simply returns
// `status`.
absl::Status MaybeWithLocation(absl::Status status, const char* file,
                               int32_t line);

// Returns a Status from either another Status, or a StatusOr.
absl::Status GetStatus(absl::Status status);

template <class T>
absl::Status GetStatus(const absl::StatusOr<T>& status) {
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
// `XLA_SHOW_CPP_ERROR_CONTEXT` is set.
absl::Status MaybeWithNewMessage(absl::Status status, const char* file,
                                 int32_t line,
                                 std::string_view new_message = "");

// Consumes the `status` and maybe throws an exception if `status` has
// a non-ok code.
//
// Ideally, this function should be used only used in the project's
// boundary, e.g. when we need to throw an exception for the user to see.
void ConsumeAndMaybeThrow(absl::Status status);

// Consumes the `status`, either returning the value it holds (for
// ok status), or throwing an exception.
template <class T>
T ConsumeAndMaybeThrow(const absl::StatusOr<T>& status) {
  ConsumeAndMaybeThrow(status.status());
  return std::move(status).value();
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_STATUS_H_
