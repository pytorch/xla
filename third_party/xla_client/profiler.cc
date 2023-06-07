#include "third_party/xla_client/profiler.h"

#include "tsl/profiler/rpc/profiler_server.h"
#include "absl/container/flat_hash_map.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/rpc/client/capture_profile.h"

namespace xla {
namespace profiler {

struct ProfilerServer::Impl {
  Impl() : server(new tsl::profiler::ProfilerServer()) {}

  std::unique_ptr<tsl::profiler::ProfilerServer> server;
};

ProfilerServer::ProfilerServer() : impl_(new Impl()) {}

void ProfilerServer::Start(int port) {
  impl_->server->StartProfilerServer(port);
}

ProfilerServer::~ProfilerServer() {}

tsl::Status Trace(
    const char* service_addr, const char* logdir, int duration_ms,
    int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options) {
  return tsl::profiler::CaptureRemoteTrace(
      service_addr, logdir, /*worker_list=*/"",
      /*include_dataset_ops=*/false, duration_ms, num_tracing_attempts,
      options);
}
}  // namespace profiler
}  // namespace xla
