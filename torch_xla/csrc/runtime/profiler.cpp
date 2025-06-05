#include "torch_xla/csrc/runtime/profiler.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/status_casters.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"

namespace torch_xla {
namespace runtime {
namespace profiler {

namespace {

const PLUGIN_Profiler_Api* FindProfilerApi(const PJRT_Api* pjrt_api) {
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(pjrt_api->extension_start);
  while (next != nullptr &&
         next->type != PJRT_Extension_Type::PJRT_Extension_Type_Profiler) {
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

const std::string TslProfilerSessionWrapper::Stop() const {
  tensorflow::profiler::XSpace xspace;
  // Disables the ProfilerSession
  xla::ThrowIfError(this->session->CollectData(&xspace));
  std::string xspace_str = xspace.SerializeAsString();
  return xspace_str;
}

void TslProfilerSessionWrapper::Export(
    const std::string& xspace_str, const std::string& tensorboard_dir) const {
  tensorflow::profiler::XSpace xspace_proto;
  xspace_proto.ParseFromString(xspace_str);
  xla::ThrowIfError(
      tsl::profiler::ExportToTensorBoard(xspace_proto, tensorboard_dir,
                                         /* also_export_trace_json= */ true));
}

std::unique_ptr<TslProfilerSessionWrapper> TslProfilerSessionWrapper::Create() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return absl::make_unique<runtime::profiler::TslProfilerSessionWrapper>(
      tsl::ProfilerSession::Create(options));
}

absl::Status Trace(
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
  if (!profiler_api) {
    TF_LOG(WARNING) << "Profiler API not found for PJRT plugin";
  }

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
