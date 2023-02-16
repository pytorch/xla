#ifndef XLA_CLIENT_SYS_UTIL_H_
#define XLA_CLIENT_SYS_UTIL_H_

#include <string>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace sys_util {

std::string GetEnvString(const char* name, const std::string& defval);

std::string GetEnvOrdinalPath(const char* name, const std::string& defval,
                              const int64_t ordinal);

std::string GetEnvOrdinalPath(
    const char* name, const std::string& defval,
    const char* ordinal_env = "XRT_SHARD_LOCAL_ORDINAL");

int64_t GetEnvInt(const char* name, int64_t defval);

double GetEnvDouble(const char* name, double defval);

bool GetEnvBool(const char* name, bool defval);

// Retrieves the current EPOCH time in nanoseconds.
int64_t NowNs();

}  // namespace sys_util
}  // namespace xla

#endif  // XLA_CLIENT_SYS_UTIL_H_
