#include "xla/xla_client/profiler.h"

#include "tensorflow/core/profiler/rpc/profiler_server.h"

namespace xla {
namespace profiler {

struct ProfilerServer::Impl {
  Impl() : server(new tensorflow::profiler::ProfilerServer()) {}

  std::unique_ptr<tensorflow::profiler::ProfilerServer> server;
};

ProfilerServer::ProfilerServer() : impl_(new Impl()) {}

void ProfilerServer::Start(int port) {
  impl_->server->StartProfilerServer(port);
}

ProfilerServer::~ProfilerServer() {}

}  // namespace profiler
}  // namespace xla
