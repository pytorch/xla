#include "third_party/xla_client/profiler.h"

#include "tensorflow/tsl/profiler/rpc/profiler_server.h"
#include "tensorflow/tsl/profiler/lib/profiler_session.h"

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


struct ProfilerSession::Impl {
  Impl() : session(new tsl::ProfilerSession()) {}

  std::unique_ptr<tsl::ProfilerSession> session;
};

ProfilerSession::ProfilerSession() : impl_(new Impl()) {}

static std::unique_ptr<ProfilerSession> Create(
    const tensorflow::ProfileOptions& options) {
void ProfilerSession::Create(int port) {
  impl_->session->Create(options);
}

ProfilerSession::~ProfilerSession() {}

tensorflow::ProfileOptions ProfilerSession::DefaultPythonProfileOptions() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return options;
}



}  // namespace profiler
}  // namespace xla
