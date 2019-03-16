#include "tensorflow/compiler/xla/xla_client/xrt_session_cache.h"

#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"

namespace xla {

XrtSessionCache::XrtSessionCache(std::function<void(XrtSession*)> initfn)
    : initfn_(std::move(initfn)) {}

XrtSessionCache::Ref XrtSessionCache::GetSession(const string& target) {
  std::lock_guard<std::mutex> lock(lock_);
  auto& session_queue = session_map_[target];
  if (!session_queue.empty()) {
    std::shared_ptr<XrtSession> session = std::move(session_queue.back());
    session_queue.pop_back();
    session->Reset();
    return Ref(this, std::move(session));
  }
  return Ref(this, CreateSession(target));
}

XrtSession* XrtSessionCache::GetSession(const string& target,
                                        SessionMap* session_map) {
  auto it = session_map->find(target);
  if (it == session_map->end()) {
    it = session_map->emplace(target, GetSession(target)).first;
  }
  return it->second.get();
}

void XrtSessionCache::AddSession(std::shared_ptr<XrtSession> session) {
  std::lock_guard<std::mutex> lock(lock_);
  session_map_[session->target()].push_back(std::move(session));
}

std::shared_ptr<XrtSession> XrtSessionCache::CreateSession(
    const string& target) const {
  XLA_COUNTER("XrtSessionCount", 1);
  tensorflow::SessionOptions session_options;
  session_options.env = tensorflow::Env::Default();
  session_options.target = target;

  string compression = sys_util::GetEnvString("XRT_GRPC_COMPRESSION", "");
  if (!compression.empty()) {
    tensorflow::RPCOptions* rpc_options =
        session_options.config.mutable_rpc_options();
    rpc_options->set_compression_algorithm(compression);
    rpc_options->set_compression_level(
        sys_util::GetEnvInt("XRT_GRPC_COMPRESSION_LEVEL", 3));
  }
  std::shared_ptr<XrtSession> session =
      std::make_shared<XrtSession>(session_options);
  if (initfn_ != nullptr) {
    initfn_(session.get());
  }
  return session;
}

}  // namespace xla
