#include "tensorflow/compiler/xla/xla_client/xrt_session_cache.h"

#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"

namespace xla {

XrtSessionCache::XrtSessionCache(tensorflow::ConfigProto config,
                                 std::function<void(XrtSession*)> initfn)
    : config_(std::move(config)), initfn_(std::move(initfn)) {}

XrtSessionCache::Ref XrtSessionCache::GetSession(const std::string& target) {
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

XrtSession* XrtSessionCache::GetSession(const std::string& target,
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
    const std::string& target) const {
  XLA_COUNTER("XrtSessionCount", 1);
  tensorflow::SessionOptions session_options;
  session_options.env = tensorflow::Env::Default();
  session_options.target = target;
  session_options.config = config_;

  tensorflow::RPCOptions* rpc_options =
      session_options.config.mutable_rpc_options();

  std::string compression = sys_util::GetEnvString("XRT_GRPC_COMPRESSION", "");
  if (!compression.empty()) {
    rpc_options->set_compression_algorithm(compression);
    rpc_options->set_compression_level(
        sys_util::GetEnvInt("XRT_GRPC_COMPRESSION_LEVEL", 3));
  }

  bool multi_stream = sys_util::GetEnvBool("XRT_GRPC_MULTISTREAM", true);
  rpc_options->set_disable_session_connection_sharing(multi_stream);

  std::shared_ptr<XrtSession> session =
      std::make_shared<XrtSession>(session_options);
  if (initfn_ != nullptr) {
    initfn_(session.get());
  }
  return session;
}

}  // namespace xla
