#include "torch_xla/csrc/runtime/profiler.h"

#include "absl/container/flat_hash_map.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/rpc/client/capture_profile.h"
#include "tsl/profiler/rpc/profiler_server.h"
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"

namespace torch_xla {
namespace runtime {
namespace profiler {

namespace {

const PLUGIN_Profiler_Api* FindProfilerApi(const PJRT_Api* pjrt_api) {
  const PJRT_Structure_Base* next =
      reinterpret_cast<const PJRT_Structure_Base*>(pjrt_api->extension_start);
  while (next != nullptr &&
         next->type != PJRT_Structure_Type::PJRT_Structure_Type_Profiler) {
    next = next->next;
  }
  if (next == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<const PJRT_Profiler_Extension*>(next)->profiler_api;
}

}  // namespace

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

void RegisterProfilerForPlugin(const PJRT_Api* c_api) {
  const PLUGIN_Profiler_Api* profiler_api = FindProfilerApi(c_api);
  XLA_CHECK(profiler_api);

  tsl::profiler::ProfilerFactory create_func =
      [profiler_api](const tensorflow::ProfileOptions& options) {
        return std::make_unique<xla::profiler::PluginTracer>(profiler_api,
                                                             options);
      };
  tsl::profiler::RegisterProfilerFactory(std::move(create_func));
}

}  // namespace profiler
}  // namespace runtime
}  // namespace torch_xla
