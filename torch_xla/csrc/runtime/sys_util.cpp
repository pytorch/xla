#include "torch_xla/csrc/runtime/sys_util.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "absl/strings/str_cat.h"

namespace torch_xla {
namespace runtime {
namespace sys_util {

std::string GetEnvString(const char* const name, const std::string& defval) {
  const char* const env = std::getenv(name);
  return env != nullptr ? env : defval;
}

std::string GetEnvOrdinalPath(const char* const name, const std::string& defval,
                              const int64_t ordinal) {
  std::string path = GetEnvString(name, defval);
  if (!path.empty()) {
    if (ordinal >= 0) {
      path = absl::StrCat(path, ".", ordinal);
    }
  }
  return path;
}

std::string GetEnvOrdinalPath(const char* const name, const std::string& defval,
                              const char* const ordinal_env) {
  return GetEnvOrdinalPath(name, defval, GetEnvInt(ordinal_env, -1));
}

int64_t GetEnvInt(const char* const name, const int64_t defval) {
  const char* const env = std::getenv(name);
  return env != nullptr ? std::atol(env) : defval;
}

double GetEnvDouble(const char* const name, const double defval) {
  const char* const env = std::getenv(name);
  return env != nullptr ? std::atof(env) : defval;
}

bool GetEnvBool(const char* const name, const bool defval) {
  const char* const env = std::getenv(name);
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

int64_t NowNs() {
  const auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

}  // namespace sys_util
}  // namespace runtime
}  // namespace torch_xla
