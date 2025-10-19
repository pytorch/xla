#include "torch_xla/csrc/status.h"

#include <c10/util/Exception.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/stacktrace.h"

namespace torch_xla {

// Indent the stack frame representation so that it's easier to see.
constexpr char kFramePrefix[] = "\n    ";

// Creates the stack frame representation for the status propagation trace
// entry.
//
// The resulting string will be appended to the existing status propagation
// trace of the status currently being processed.
//
// Example:
//   \n    From: <function> at <file>:<line> [(error: <message>)]
//
static std::string GetStackFrame(const char* file, const int32_t line,
                                 const char* function,
                                 const std::string_view new_message) {
  auto error_suffix =
      new_message.empty() ? "" : absl::StrCat(" (error: ", new_message, ")");
  return absl::StrCat(kFramePrefix, "From: ", function, " at ", file, ":", line,
                      error_suffix);
}

// Convenient function that retrieves the status propagation trace payload
// if it exists. Otherwise, returns an empty absl::Cord.
static absl::Cord GetStatusPropagationTraceOrEmpty(const absl::Status& status) {
  auto opt = status.GetPayload(kStatusPropagationTraceKey);
  return opt.has_value() ? *opt : absl::Cord();
}

absl::Status status_internal::MaybeWithLocation(const absl::Status& status,
                                                const char* file,
                                                const int32_t line,
                                                const char* function) {
  ABSL_CHECK(!status.ok());

  // Return the same status if we don't need to add the C++ source location.
  if (!torch::get_cpp_stacktraces_enabled()) {
    return status;
  }

  // Make sure this is only called on fresh `status` instances.
  ABSL_CHECK(GetStatusPropagationTraceOrEmpty(status).empty());

  // Adding source location to `status` has the same semantics as overwriting
  // the status message:
  //   1. An stack frame will be added to the status propagation trace
  //   2. The status' message will be the same
  return MaybeWithNewMessage(status, file, line, function, status.message());
}

absl::Status status_internal::MaybeWithNewMessage(
    const absl::Status& status, const char* file, const int32_t line,
    const char* function, const std::string_view new_message) {
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
  //
  //     1. append the current stack frame to the status propagation trace
  //        payload
  //
  //     2. append the new error message, if not empty
  if (torch::get_cpp_stacktraces_enabled()) {
    auto status_propagation_trace = GetStatusPropagationTraceOrEmpty(status);
    status_propagation_trace.Append(
        GetStackFrame(file, line, function, new_message));
    new_status.SetPayload(kStatusPropagationTraceKey, status_propagation_trace);
  }

  return new_status;
}

// Get a formatted string representation of the status propagation trace
// if it's not empty.
static std::string GetFormattedStatusPropagationTrace(
    const absl::Status& status) {
  auto status_propagation_trace = GetStatusPropagationTraceOrEmpty(status);
  return status_propagation_trace.empty()
             ? ""
             : absl::StrCat("\n\nStatus Propagation Trace:",
                            status_propagation_trace.Flatten());
}

std::string BuildStatusErrorMessage(const absl::Status& status) {
  return absl::StrCat(status.message(),
                      GetFormattedStatusPropagationTrace(status));
}

// Return a line break if torch::get_cpp_stacktraces_enabled() is true.
static std::string LineBreakIfCppStacktracesEnabled() {
  return torch::get_cpp_stacktraces_enabled() ? "\n" : "";
}

void status_internal::ThrowStatusError(const absl::Status& status,
                                       const char* file, const int32_t line,
                                       const char* function,
                                       std::string_view message) {
  ABSL_CHECK(!status.ok());
  absl::Status new_status = status_internal::MaybeWithNewMessage(
      status, file, line, function, message);
  TORCH_CHECK(false, absl::StrCat(BuildStatusErrorMessage(new_status),
                                  LineBreakIfCppStacktracesEnabled()));
}

void status_internal::OkOrDie(const absl::Status& status, const char* file,
                              const int32_t line, const char* function,
                              std::string_view message) {
  if (status.ok()) {
    return;
  }

  std::ostringstream oss;
  oss << "\n\n"
      << "Internal Error:\n";

  if (!message.empty()) {
    oss << "    " << message << "\n";
  }

  oss << "    This is a bug! Please, open an issue in the PyTorch/XLA "
      << "GitHub repository: https://github.com/pytorch/xla"
      << "\n\n"
      << "Status Error:\n"
      << "    "
      << BuildStatusErrorMessage(
             status_internal::MaybeWithNewMessage(status, file, line, function))
      << "\n";

  ABSL_CHECK(status.ok()) << oss.str();
}

}  // namespace torch_xla
