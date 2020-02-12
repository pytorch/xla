#ifndef XLA_CLIENT_DEBUG_MACROS_H_
#define XLA_CLIENT_DEBUG_MACROS_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#define XLA_ERROR() TF_ERROR_STREAM()
#define XLA_CHECK(c) TF_CHECK(c) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_OK(c) \
  TF_CHECK_OK(c) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) \
  TF_CHECK_EQ(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) \
  TF_CHECK_NE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) \
  TF_CHECK_LE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) \
  TF_CHECK_GE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) \
  TF_CHECK_LT(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) \
  TF_CHECK_GT(a, b) << "\n" << tensorflow::CurrentStackTrace()

template <typename T>
T ConsumeValue(xla::StatusOr<T>&& status) {
  XLA_CHECK_OK(status.status());
  return status.ConsumeValueOrDie();
}

#endif  // XLA_CLIENT_DEBUG_MACROS_H_
