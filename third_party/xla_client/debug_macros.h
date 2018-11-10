#ifndef TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_
#define TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#define XLA_CHECK(c) CHECK(c) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) CHECK_EQ(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) CHECK_NE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) CHECK_LE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) CHECK_GE(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) CHECK_LT(a, b) << "\n" << tensorflow::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) CHECK_GT(a, b) << "\n" << tensorflow::CurrentStackTrace()

#endif  // TENSORFLOW_COMPILER_XLA_RPC_DEBUG_MACROS_H_
