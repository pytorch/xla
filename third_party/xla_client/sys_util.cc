#include "tensorflow/compiler/xla/xla_client/sys_util.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace xla {
namespace sys_util {

string GetEnvString(const char* name, const string& defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? env : defval;
}

int64 GetEnvInt(const char* name, int64 defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atol(env) : defval;
}

double GetEnvDouble(const char* name, double defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atof(env) : defval;
}

bool GetEnvBool(const char* name, bool defval) {
  const char* env = std::getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  return std::atoi(env) != 0;
}

int64 NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

}  // namespace sys_util
}  // namespace xla
