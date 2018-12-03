#ifndef TENSORFLOW_COMPILER_XLA_RPC_TF_LOGGING_H_
#define TENSORFLOW_COMPILER_XLA_RPC_TF_LOGGING_H_

#include "tensorflow/core/platform/logging.h"

// It happens that Caffe defined the same exact Google macros, hiding the TF
// ones, and making log messages disappear.
// Unfortunately to get those back, we have to poke through the TF
// implementaiton of them.
#define TF_LOG(severity) _TF_LOG_##severity

#define TF_VLOG_IS_ON(lvl)                                                  \
  (([](int level, const char* fname) {                                      \
    static const bool vmodule_activated =                                   \
        ::tensorflow::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                               \
  })(lvl, __FILE__))

#define TF_VLOG(level)                                           \
  TF_PREDICT_TRUE(!TF_VLOG_IS_ON(level))                         \
  ? (void)0                                                      \
  : ::tensorflow::internal::Voidifier() &                        \
          ::tensorflow::internal::LogMessage(__FILE__, __LINE__, \
                                             tensorflow::INFO)

#define TF_CHECK(condition)           \
  if (TF_PREDICT_FALSE(!(condition))) \
  TF_LOG(FATAL) << "Check failed: " #condition " "

#define TF_CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// TF_CHECK_EQ/NE/...
#define TF_CHECK_EQ(val1, val2) TF_CHECK_OP(Check_EQ, ==, val1, val2)
#define TF_CHECK_NE(val1, val2) TF_CHECK_OP(Check_NE, !=, val1, val2)
#define TF_CHECK_LE(val1, val2) TF_CHECK_OP(Check_LE, <=, val1, val2)
#define TF_CHECK_LT(val1, val2) TF_CHECK_OP(Check_LT, <, val1, val2)
#define TF_CHECK_GE(val1, val2) TF_CHECK_OP(Check_GE, >=, val1, val2)
#define TF_CHECK_GT(val1, val2) TF_CHECK_OP(Check_GT, >, val1, val2)
#define TF_CHECK_NOTNULL(val)                              \
  ::tensorflow::internal::CheckNotNull(__FILE__, __LINE__, \
                                       "'" #val "' Must be non NULL", (val))

#endif  // TENSORFLOW_COMPILER_XLA_RPC_TF_LOGGING_H_
