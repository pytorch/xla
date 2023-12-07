#ifndef XLA_CLIENT_PROFILER_H_
#define XLA_CLIENT_PROFILER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tsl/platform/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace torch_xla {
namespace runtime {
namespace profiler {

class ProfilerServer {
  struct Impl;

 public:
  ProfilerServer();
  ~ProfilerServer();
  void Start(int port);

 private:
  std::unique_ptr<Impl> impl_;
};

tsl::Status Trace(
    const char* service_addr, const char* logdir, int duration_ms,
    int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options);

void RegisterProfilerForPlugin(const PJRT_Api* c_api);

}  // namespace profiler
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_PROFILER_H_
