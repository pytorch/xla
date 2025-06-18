#include "torch_xla/csrc/status.h"

#include "absl/log/absl_check.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {

bool showCppErrorContext() {
  return runtime::sys_util::GetEnvBool(runtime::env::kEnvShowCppErrorContext,
                                       false);
}

// Common function for generating file location information with a space in the
// beginning.
static std::string LocationStrWithSpace(const char* file, const int32_t line) {
  return absl::StrCat(" (at ", file, ":", line, ")");
}

absl::Status MaybeWithLocation(const absl::Status& status, const char* file,
                               const int32_t line) {
  ABSL_CHECK(!status.ok());
  std::string_view message = status.message();
  return absl::Status(
      status.code(),
      (showCppErrorContext())
          ? absl::StrCat(message, LocationStrWithSpace(file, line))
          : message);
}

absl::Status MaybeWithNewMessage(const absl::Status& status, const char* file,
                                 const int32_t line,
                                 const std::string_view new_message) {
  ABSL_CHECK(!status.ok());
  std::string_view old_message = status.message();

  // Replace the old status message with `new_message`, if it's not empty.
  //
  // The idea is that whenever `new_message` is given, it should have more
  // context to give a better error message to the user.
  std::string_view message = (new_message.empty()) ? old_message : new_message;

  // If `kEnvShowCppErrorContext` is set, show the context of this error.
  // In other words, show:
  //   - The error location
  //   - The old messages that were replaced by `new_message`.
  //
  // This should give more context for developers. Showing the older error
  // messages alongside their debug information.
  std::string context;
  if (showCppErrorContext()) {
    context = LocationStrWithSpace(file, line);
    if (!new_message.empty()) {
      context = absl::StrCat(context, "\nFrom Error: ", old_message);
    }
  }

  return absl::Status(status.code(), absl::StrCat(message, context));
}

void ConsumeAndMaybeThrow(absl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

}  // namespace torch_xla
