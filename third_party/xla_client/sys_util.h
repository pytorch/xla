#ifndef TENSORFLOW_COMPILER_XLA_RPC_SYS_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_RPC_SYS_UTIL_H_

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace sys_util {

string GetEnvString(const char* name, const string& defval);

int64 GetEnvInt(const char* name, int64 defval);

double GetEnvDouble(const char* name, double defval);

bool GetEnvBool(const char* name, bool defval);

// Retrieves the current EPOCH time in nanoseconds.
int64 NowNs();

}  // namespace sys_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_SYS_UTIL_H_
