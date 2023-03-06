#ifndef XLA_CLIENT_DEBUG_MACROS_H_
#define XLA_CLIENT_DEBUG_MACROS_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/stacktrace.h"
#include "third_party/xla_client/tf_logging.h"

#define XLA_ERROR() TF_ERROR_STREAM()
#define XLA_CHECK(c) TF_CHECK(c) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_OK(c) TF_CHECK_OK(c) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) TF_CHECK_EQ(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) TF_CHECK_NE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) TF_CHECK_LE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) TF_CHECK_GE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) TF_CHECK_LT(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) TF_CHECK_GT(a, b) << "\n" << tsl::CurrentStackTrace()

template <typename T>
T ConsumeValue(xla::StatusOr<T>&& status) {
  XLA_CHECK_OK(status.status());
  return std::move(status).value();
}

#endif  // XLA_CLIENT_DEBUG_MACROS_H_
