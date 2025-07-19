#include "torch_xla/csrc/status.h"

#include <torch/csrc/utils/cpp_stacktraces.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "absl/log/absl_check.h"
#include "tsl/platform/stacktrace.h"

namespace torch_xla {

// Indent the stacktrace so that it's easier to see.
constexpr char kEntryPrefix[] = "\n    ";

// Creates an error propagation stacktrace entry.
//
// The resulting entry will be appended to the existing stacktrace of the status
// currently being processed.
//
// Example:
//   From: <file>:<line> [(error: <message>)]
//
static std::string GetStacktraceEntry(const char* file, const int32_t line,
                                      const std::string_view new_message) {
  auto error_suffix =
      new_message.empty() ? "" : absl::StrCat(" (error: ", new_message, ")");
  return absl::StrCat(kEntryPrefix, "From: ", file, ":", line, error_suffix);
}

// Convenient function that retrieves the stacktrace payload if it exists.
// Otherwise, returns an empty absl::Cord.
static absl::Cord GetStacktraceOrEmptyCord(const absl::Status& status) {
  auto opt = status.GetPayload(kStacktraceKey);
  return opt.has_value() ? *opt : absl::Cord();
}

absl::Status status_internal::MaybeWithLocation(const absl::Status& status,
                                                const char* file,
                                                const int32_t line) {
  ABSL_CHECK(!status.ok());

  // Return the same status if we don't need to add the C++ source location.
  if (!torch::get_cpp_stacktraces_enabled()) {
    return status;
  }

  // Make sure this is only called on fresh `status` instances.
  ABSL_CHECK(GetStacktraceOrEmptyCord(status).empty());

  // Adding source location to `status` has the same semantics as overwriting
  // the status message:
  //   1. An stacktrace entry will be added
  //   2. The status' message will be the same
  return MaybeWithNewMessage(status, file, line, status.message());
}

absl::Status status_internal::MaybeWithNewMessage(
    const absl::Status& status, const char* file, const int32_t line,
    const std::string_view new_message) {
  ABSL_CHECK(!status.ok());

  // Return the same status if:
  //   1. we don't need to add the C++ source location.
  //   2. there's no new message to replace the old one.
  if (!torch::get_cpp_stacktraces_enabled() && new_message.empty()) {
    return status;
  }

  // Replace the old status message with `new_message`, if it's not empty.
  //
  // The idea is that whenever `new_message` is given, it should have more
  // context to give a better error message to the user.
  auto new_status = absl::Status(
      status.code(), new_message.empty() ? status.message() : new_message);

  // If `TORCH_SHOW_CPP_STACKTRACES` is set:
  //     1. append the current source location to the stacktrace payload
  //     2. append the new error message, if not empty
  if (torch::get_cpp_stacktraces_enabled()) {
    auto new_stacktrace = GetStacktraceOrEmptyCord(status);
    new_stacktrace.Append(GetStacktraceEntry(file, line, new_message));
    new_status.SetPayload(kStacktraceKey, new_stacktrace);
  }

  return new_status;
}

void MaybeThrow(const absl::Status& status) {
  if (!status.ok()) {
    auto status_stacktrace = GetStacktraceOrEmptyCord(status);
    auto status_stacktrace_str =
        status_stacktrace.empty()
            ? ""
            : absl::StrCat("\n\nStatus Propagation Stacktrace:",
                           status_stacktrace.Flatten());
    auto cpp_stacktrace_str =
        torch::get_cpp_stacktraces_enabled()
            ? absl::StrCat("\n\nC++ Stacktrace:\n", tsl::CurrentStackTrace())
            : "";
    throw std::runtime_error(absl::StrCat(
        status.message(), status_stacktrace_str, cpp_stacktrace_str));
  }
}

}  // namespace torch_xla
