#include "torch_xla/csrc/status.h"

#include "absl/log/absl_check.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {

// Returns whether we should show C++ error context.
//
// More specifically, whether the `XLA_SHOW_CPP_ERROR_CONTEXT` environment
// variable is set or not.
static bool ShouldShowCppErrorContext() {
  static const bool show_cpp_error_context = runtime::sys_util::GetEnvBool(
      runtime::env::kEnvShowCppErrorContext, false);
  return show_cpp_error_context;
}

// Common function for generating file location information with a space in the
// beginning.
static std::string LocationStrWithSpace(const char* file, const int32_t line) {
  return absl::StrCat(" (at ", file, ":", line, ")");
}

absl::Status MaybeWithLocation(const absl::Status& status, const char* file,
                               const int32_t line) {
  ABSL_CHECK(!status.ok());

  // Return the same status if we don't need to add the C++ source location.
  if (!ShouldShowCppErrorContext()) {
    return status;
  }

  return absl::Status(
      status.code(),
      absl::StrCat(status.message(), LocationStrWithSpace(file, line)));
}

absl::Status MaybeWithNewMessage(const absl::Status& status, const char* file,
                                 const int32_t line,
                                 const std::string_view new_message) {
  ABSL_CHECK(!status.ok());

  // Return the same status if:
  //   1. we don't need to add the C++ source location.
  //   2. there's no new message to replace the old one.
  if (!ShouldShowCppErrorContext() && new_message.empty()) {
    return status;
  }

  std::string_view old_message = status.message();

  // Replace the old status message with `new_message`, if it's not empty.
  //
  // The idea is that whenever `new_message` is given, it should have more
  // context to give a better error message to the user.
  std::string_view message = new_message.empty() ? old_message : new_message;

  // If `XLA_SHOW_CPP_ERROR_CONTEXT` is set, show the context of this error.
  // In other words, show:
  //   - The error location
  //   - The old messages that were replaced by `new_message`.
  //
  // This should give more context for developers. Showing the older error
  // messages alongside their debug information.
  std::string context;
  if (ShouldShowCppErrorContext()) {
    context = LocationStrWithSpace(file, line);
    if (!new_message.empty()) {
      context = absl::StrCat(context, "\nFrom Error: ", old_message);
    }
  }

  return absl::Status(status.code(), absl::StrCat(message, context));
}

void MaybeThrow(const absl::Status& status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

}  // namespace torch_xla
