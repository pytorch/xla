#ifndef XLA_TORCH_XLA_CSRC_STATUS_H_
#define XLA_TORCH_XLA_CSRC_STATUS_H_

#include "absl/status/statusor.h"

namespace torch_xla {

// Creates a new Status instance, appending the current location (e.g. file and
// line information) to the status message.
//
// This should be used whenever we are returning new error status, instead of
// propagating. Then, if `kEnvShowCppErrorContext` environment variable is set,
// the location information will be shown.
#define XLA_ERROR_WITH_LOCATION(STATUS) \
  ::torch_xla::MaybeWithLocation(std::move(STATUS), __FILE__, __LINE__)

#define _XLA_RETURN_IF_ERROR(EXPR, THEN, ...)                         \
  do {                                                                \
    auto _status_var = (EXPR);                                        \
    if (!_status_var.ok()) {                                          \
      return ::torch_xla::MaybeWithNewMessage(                        \
          std::move(_status_var), __FILE__, __LINE__, ##__VA_ARGS__); \
    }                                                                 \
    THEN;                                                             \
  } while (0)

// Propagates `REXPR`, in case it's a non-ok status.
#define XLA_RETURN_IF_ERROR(REXPR, ...) \
  _XLA_RETURN_IF_ERROR(REXPR, {}, ##__VA_ARGS__)

// Propagates `REXPR`, in case it's a non-ok status. Otherwise, assign
// its result to `LHS`.
#define XLA_ASSIGN_OR_RETURN(LHS, REXPR, ...) \
  _XLA_RETURN_IF_ERROR(                       \
      REXPR, { LHS = std::move(_status_var.value()); }, ##__VA_ARGS__)

// Maybe shows location information in the status message.
//
// This function assumes that `status` is a non-ok status.
absl::Status MaybeWithLocation(absl::Status&& status, const char* file,
                               const int32_t line);

// Maybe replace the current `status` message with `new_message`.
//
// This function assumes that `status` is a non-ok status.
//
// Rationale: if given, `new_message` has more context, which makes it possible
// for better error messages to the user.
//
// This function also appends file location information to the error message, if
// `kEnvShowCppErrorContext` is set.
absl::Status MaybeWithNewMessage(absl::Status&& status, const char* file,
                                 const int32_t line,
                                 const std::string_view new_message = "");

template <class T>
absl::Status MaybeWithNewMessage(absl::StatusOr<T>&& status, const char* file,
                                 const int32_t line,
                                 const std::string_view new_message = "") {
  return MaybeWithNewMessage(std::move(status).status(), file, line,
                             new_message);
}

// Consumes the `status` and maybe throws an exception if `status` has
// a non-ok code.
//
// Ideally, this function should be used only used in the project's
// boundary, e.g. when we need to throw an exception for the user to see.
void ConsumeAndMaybeThrow(absl::Status&& status);

// Consumes the `status`, either returning the value it holds (for
// ok status), or throwing an exception.
template <class T>
T ConsumeAndMaybeThrow(absl::StatusOr<T>&& status) {
  ConsumeAndMaybeThrow(std::move(status).status());
  return std::move(status.value());
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_STATUS_H_
