#ifndef TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_
#define TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_

#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#define XLA_CHECK(c) TF_CHECK(c) << "\n" << tensorflow::CurrentStackTrace()
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

#endif  // TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_
