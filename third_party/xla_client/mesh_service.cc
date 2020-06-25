#include "tensorflow/compiler/xla/xla_client/mesh_service.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.grpc.pb.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/nccl_distributed.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace xla {
namespace service {
namespace {

#define GRPC_CHECK_OK(expr)   \
  do {                        \
    auto s = expr;            \
    if (!s.ok()) {            \
      return ToGrpcStatus(s); \
    }                         \
  } while (0)

::grpc::Status ToGrpcStatus(const Status& status) {
  return status.ok()
             ? ::grpc::Status::OK
             : ::grpc::Status(static_cast<::grpc::StatusCode>(status.code()),
                              status.error_message());
}

std::ostream& operator<<(std::ostream& ostrm, const ::grpc::Status& status) {
  if (status.ok()) {
    ostrm << "OK";
  } else {
    ostrm << status.error_message() << " ("
          << static_cast<int>(status.error_code()) << ")";
  }
  return ostrm;
}

class MeshServiceImpl : public grpc::MeshService::Service {
 public:
  MeshServiceImpl(grpc::Config config) : config_(std::move(config)) {}

  ::grpc::Status GetConfig(::grpc::ServerContext* context,
                           const grpc::GetConfigRequest* request,
                           grpc::GetConfigResponse* response) override;

  ::grpc::Status Rendezvous(::grpc::ServerContext* context,
                            const grpc::RendezvousRequest* request,
                            grpc::RendezvousResponse* response) override;

  ::grpc::Status GetNcclUniqueUid(
      ::grpc::ServerContext* context,
      const grpc::GetNcclUniqueUidRequest* request,
      grpc::GetNcclUniqueUidResponse* response) override;

 private:
  class RendezvousData {
   public:
    explicit RendezvousData(size_t count, const std::set<int64>& replicas)
        : count_(count),
          replicas_(replicas),
          mwait_(count),
          release_count_(0) {}

    bool Release() { return release_count_.fetch_add(1) == 0; }

    ::grpc::Status Wait() {
      ::grpc::Status status =
          ToGrpcStatus(xla::util::CheckedCall([&]() { mwait_.Wait(); }));
      if (status.ok()) {
        std::lock_guard<std::mutex> lock(lock_);
        status = status_;
      }
      return status;
    }

    void Complete(int64 ordinal, std::string payload,
                  const std::set<int64>& replicas) {
      std::lock_guard<std::mutex> lock(lock_);
      if ((!replicas_.empty() && replicas_.count(ordinal) == 0) ||
          (replicas_.empty() && ordinal >= count_)) {
        status_ = ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                 absl::StrCat("Invalid ordinal: ", ordinal));
      } else if (replicas != replicas_) {
        status_ = ::grpc::Status(
            ::grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrCat("Mismatching replicas: (",
                         absl::StrJoin(replicas_, ", "), ") vs. (",
                         absl::StrJoin(replicas, ", "), ")"));
      } else {
        auto insert_result = payloads_.emplace(ordinal, std::move(payload));
        if (!insert_result.second) {
          status_ =
              ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                             absl::StrCat("Duplicate ordinal: ", ordinal));
        }
      }
      mwait_.Done();
    }

    const std::map<int64, std::string>& Payloads() const { return payloads_; };

   private:
    size_t count_;
    std::set<int64> replicas_;
    std::mutex lock_;
    util::MultiWait mwait_;
    std::atomic<size_t> release_count_;
    std::map<int64, std::string> payloads_;
    ::grpc::Status status_;
  };

  std::shared_ptr<RendezvousData> GetRendezvous(
      const std::string& tag, const std::set<int64>& replicas) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = rendezvous_map_.find(tag);
    if (it == rendezvous_map_.end()) {
      size_t count = replicas.empty() ? config_.mesh_size() : replicas.size();
      it = rendezvous_map_
               .emplace(tag, std::make_shared<RendezvousData>(count, replicas))
               .first;
    }
    return it->second;
  }

  void ReleaseRendezvous(const std::string& tag,
                         const std::shared_ptr<RendezvousData>& rendezvous) {
    if (rendezvous->Release()) {
      std::lock_guard<std::mutex> lock(lock_);
      rendezvous_map_.erase(tag);
    }
  }

  std::mutex lock_;
  grpc::Config config_;
  std::unordered_map<std::string, std::shared_ptr<RendezvousData>>
      rendezvous_map_;
};

::grpc::Status MeshServiceImpl::GetConfig(::grpc::ServerContext* context,
                                          const grpc::GetConfigRequest* request,
                                          grpc::GetConfigResponse* response) {
  TF_VLOG(3) << "Got config fetch request: peer=" << context->peer();
  response->mutable_config()->CopyFrom(config_);
  return ::grpc::Status::OK;
}

::grpc::Status MeshServiceImpl::Rendezvous(
    ::grpc::ServerContext* context, const grpc::RendezvousRequest* request,
    grpc::RendezvousResponse* response) {
  std::set<int64> replicas(request->replicas().begin(),
                           request->replicas().end());
  auto rendezvous = GetRendezvous(request->tag(), replicas);
  rendezvous->Complete(request->ordinal(), request->payload(), replicas);
  TF_VLOG(3) << "Entering rendezvous: ordinal=" << request->ordinal()
             << ", tag=" << request->tag() << ", peer=" << context->peer();
  ::grpc::Status status = rendezvous->Wait();
  TF_VLOG(3) << "Exiting rendezvous: ordinal=" << request->ordinal()
             << ", tag=" << request->tag() << ", peer=" << context->peer()
             << ", status=" << status;
  if (status.ok()) {
    for (auto& ordinal_payload : rendezvous->Payloads()) {
      response->add_payloads(ordinal_payload.second);
    }
  }
  ReleaseRendezvous(request->tag(), rendezvous);
  return status;
}

::grpc::Status MeshServiceImpl::GetNcclUniqueUid(
    ::grpc::ServerContext* context,
    const grpc::GetNcclUniqueUidRequest* request,
    grpc::GetNcclUniqueUidResponse* response) {
  std::vector<int64> replicas;
  for (auto& replica : request->replicas()) {
    replicas.push_back(replica);
  }
  TF_VLOG(3) << "Got NCCL UID fetch request: replicas=("
             << absl::StrJoin(replicas, ", ") << "), peer=" << context->peer();
  response->set_uid(nccl_detail::GetNcclUniqueUid(replicas));
  return ::grpc::Status::OK;
}

}  // namespace

struct MeshService::Impl {
  Impl(const std::string& address, grpc::Config config)
      : impl(std::move(config)) {
    ::grpc::ServerBuilder builder;
    int64 max_msg_size =
        sys_util::GetEnvInt("XRT_MESH_MAX_MSGSIZE", 1024 * 1024 * 1024);
    builder.SetMaxReceiveMessageSize(max_msg_size);
    builder.SetMaxSendMessageSize(max_msg_size);
    builder.AddListeningPort(address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&impl);
    server = builder.BuildAndStart();
  }

  MeshServiceImpl impl;
  std::unique_ptr<::grpc::Server> server;
};

MeshService::MeshService(const std::string& address, grpc::Config config)
    : impl_(new Impl(address, std::move(config))) {}

MeshService::~MeshService() {}

void MeshService::Shutdown() {
  impl_->server->Shutdown();
  impl_->server->Wait();
}

struct MeshClient::Impl {
  explicit Impl(const std::string& address) : address(address) {
    channel =
        ::grpc::CreateChannel(address, ::grpc::InsecureChannelCredentials());
    stub = grpc::MeshService::NewStub(channel);
  }

  std::shared_ptr<::grpc::Channel> channel;
  std::unique_ptr<grpc::MeshService::Stub> stub;
  std::string address;
};

MeshClient* MeshClient::Get() {
  auto create_client = []() {
    std::string mesh_service_address =
        sys_util::GetEnvString("XRT_MESH_SERVICE_ADDRESS", "");
    return !mesh_service_address.empty() ? new MeshClient(mesh_service_address)
                                         : nullptr;
  };
  static MeshClient* client = create_client();
  return client;
}

MeshClient::MeshClient(const std::string& address) : impl_(new Impl(address)) {
  int64 connect_wait_seconds =
      sys_util::GetEnvInt("XRT_MESH_CONNECT_WAIT", 300);
  TF_LOG(INFO) << "Waiting to connect to client mesh master ("
               << connect_wait_seconds << " seconds) " << address;
  XLA_CHECK(impl_->channel->WaitForConnected(
      std::chrono::system_clock::now() +
      std::chrono::seconds(connect_wait_seconds)))
      << "Failed to connect to client mesh master: " << address;
}

MeshClient::~MeshClient() {}

const std::string& MeshClient::address() const { return impl_->address; }

grpc::Config MeshClient::GetConfig() const {
  ::grpc::ClientContext context;
  grpc::GetConfigRequest request;
  grpc::GetConfigResponse response;
  ::grpc::Status status = impl_->stub->GetConfig(&context, request, &response);
  if (!status.ok()) {
    XLA_ERROR() << "Failed to retrieve mesh configuration: " << status;
  }
  return std::move(*response.mutable_config());
}

std::vector<std::string> MeshClient::Rendezvous(
    int ordinal, const std::string& tag, const std::string& payload,
    absl::Span<const int64> replicas) const {
  ::grpc::ClientContext context;
  grpc::RendezvousRequest request;
  grpc::RendezvousResponse response;
  request.set_tag(tag);
  request.set_payload(payload);
  request.set_ordinal(ordinal);
  for (auto& replica : replicas) {
    request.add_replicas(replica);
  }
  TF_VLOG(3) << "Waiting for rendezvous: ordinal=" << ordinal << " tag=" << tag;
  ::grpc::Status status = impl_->stub->Rendezvous(&context, request, &response);
  TF_VLOG(3) << "Rendezvous wait complete: " << tag;
  if (!status.ok()) {
    XLA_ERROR() << "Failed to meet rendezvous '" << tag << "': " << status;
  }
  std::vector<std::string> rv_payloads;
  for (auto& rv_payload : response.payloads()) {
    rv_payloads.push_back(rv_payload);
  }
  return rv_payloads;
}

std::string MeshClient::GetNcclUniqueUid(
    absl::Span<const int64> replicas) const {
  ::grpc::ClientContext context;
  grpc::GetNcclUniqueUidRequest request;
  grpc::GetNcclUniqueUidResponse response;
  for (auto& replica : replicas) {
    request.add_replicas(replica);
  }

  TF_VLOG(3) << "Waiting for NCCL UID: replicas=("
             << absl::StrJoin(replicas, ", ") << ")";
  ::grpc::Status status =
      impl_->stub->GetNcclUniqueUid(&context, request, &response);
  TF_VLOG(3) << "NCCL UID wait complete: " << absl::StrJoin(replicas, ", ")
             << ")";
  if (!status.ok()) {
    XLA_ERROR() << "Failed to get NCCL UID (" << absl::StrJoin(replicas, ", ")
                << "): " << status;
  }
  return response.uid();
}

}  // namespace service
}  // namespace xla
