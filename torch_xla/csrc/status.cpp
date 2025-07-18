#include "torch_xla/csrc/status.h"

#include <torch/csrc/utils/cpp_stacktraces.h>

#include "absl/log/absl_check.h"

namespace torch_xla {

// Common function for generating file location information with a space in the
// beginning.
static std::string LocationStrWithSpace(const char* file, const int32_t line) {
  return absl::StrCat(" (at ", file, ":", line, ")");
}

absl::Status MaybeWithLocation(const absl::Status& status, const char* file,
                               const int32_t line) {
  ABSL_CHECK(!status.ok());

  // Return the same status if we don't need to add the C++ source location.
  if (!torch::get_cpp_stacktraces_enabled()) {
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
  if (!torch::get_cpp_stacktraces_enabled() && new_message.empty()) {
    return status;
  }

  std::string_view old_message = status.message();

  // Replace the old status message with `new_message`, if it's not empty.
  //
  // The idea is that whenever `new_message` is given, it should have more
  // context to give a better error message to the user.
  std::string_view message = new_message.empty() ? old_message : new_message;

  // If `TORCH_SHOW_CPP_STACKTRACES` is set, show the context of this error.
  // In other words, show:
  //   1. The error location
  //   2. The old messages that were replaced by `new_message`.
  //
  // This should give more context for developers. Showing the older error
  // messages alongside their debug information.
  //
  // Note that we also condition showing source location information by (2)
  // (i.e. `new_message` is not empty) because we don't really wish to show
  // a stacktrace. Instead, we show only the history of error messages that
  // has led to the current error.
  const std::string context =
      (torch::get_cpp_stacktraces_enabled() && !new_message.empty())
          ? absl::StrCat(LocationStrWithSpace(file, line),
                         "\nFrom Error: ", old_message)
          : "";

  return absl::Status(status.code(), absl::StrCat(message, context));
}

void MaybeThrow(const absl::Status& status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

}  // namespace torch_xla
