#ifndef XLA_CLIENT_TF_LOGGING_H_
#define XLA_CLIENT_TF_LOGGING_H_

#include <sstream>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace internal {

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

struct ErrorSink : public std::basic_ostringstream<char> {};

class ErrorGenerator {
 public:
  ErrorGenerator(const char* file, int line) : file_(file), line_(line) {}

  // Use a dummy & operator as it has lower precedence WRT the streaming
  // operator, and hence allows collecting user error messages before we finally
  // throw.
  TF_ATTRIBUTE_NORETURN void operator&(
      const std::basic_ostream<char>& oss) const;

 private:
  const char* file_ = nullptr;
  int line_ = 0;
};

#define TF_ERROR_STREAM()                               \
  ::xla::internal::ErrorGenerator(__FILE__, __LINE__) & \
      ::xla::internal::ErrorSink()

#define TF_CHECK(condition)              \
  while (TF_PREDICT_FALSE(!(condition))) \
  TF_ERROR_STREAM() << "Check failed: " #condition " "

#define TF_CHECK_OP_LOG(name, op, val1, val2)                         \
  while (::tensorflow::internal::CheckOpString _result =              \
             ::tensorflow::internal::name##Impl(                      \
                 ::tensorflow::internal::GetReferenceableValue(val1), \
                 ::tensorflow::internal::GetReferenceableValue(val2), \
                 #val1 " " #op " " #val2))                            \
  TF_ERROR_STREAM() << *(_result.str_)

#define TF_CHECK_OP(name, op, val1, val2) TF_CHECK_OP_LOG(name, op, val1, val2)

// TF_CHECK_EQ/NE/...
#define TF_CHECK_EQ(val1, val2) TF_CHECK_OP(Check_EQ, ==, val1, val2)
#define TF_CHECK_NE(val1, val2) TF_CHECK_OP(Check_NE, !=, val1, val2)
#define TF_CHECK_LE(val1, val2) TF_CHECK_OP(Check_LE, <=, val1, val2)
#define TF_CHECK_LT(val1, val2) TF_CHECK_OP(Check_LT, <, val1, val2)
#define TF_CHECK_GE(val1, val2) TF_CHECK_OP(Check_GE, >=, val1, val2)
#define TF_CHECK_GT(val1, val2) TF_CHECK_OP(Check_GT, >, val1, val2)

#undef TF_CHECK_OK
#define TF_CHECK_OK(val) TF_CHECK_EQ(val, ::tensorflow::Status::OK())
#define TF_CHECK_NOTNULL(val) TF_CHECK(val != nullptr)

}  // namespace internal
}  // namespace xla

#endif  // XLA_CLIENT_TF_LOGGING_H_
