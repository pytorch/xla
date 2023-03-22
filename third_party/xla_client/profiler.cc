#include "third_party/xla_client/profiler.h"

#include "tensorflow/tsl/profiler/rpc/profiler_server.h"

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

}  // namespace profiler
}  // namespace xla
