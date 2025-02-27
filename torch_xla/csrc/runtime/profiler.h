#ifndef XLA_CLIENT_PROFILER_H_
#define XLA_CLIENT_PROFILER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tsl/profiler/lib/profiler_session.h"
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

// Profiler session implementation is based on OpenXLA, we cannot reuse
// the Python binding since it's using nanobind and torch_xla is using pybind11.
// https://github.com/openxla/xla/blob/main/xla/python/profiler.cc
class TslProfilerSessionWrapper {
 public:
  static std::unique_ptr<TslProfilerSessionWrapper> Create();

  explicit TslProfilerSessionWrapper(
      std::unique_ptr<tsl::ProfilerSession> session)
      : session(std::move(session)) {}

  void Export(const std::string& xspace_str,
              const std::string& tensorboard_dir) const;
  const std::string Stop() const;

 private:
  std::unique_ptr<tsl::ProfilerSession> session;
};

absl::Status Trace(
    const char* service_addr, const char* logdir, int duration_ms,
    int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options);

void RegisterProfilerForPlugin(const PJRT_Api* c_api);

}  // namespace profiler
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_PROFILER_H_
