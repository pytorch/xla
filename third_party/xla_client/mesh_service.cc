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
#include <mutex>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.grpc.pb.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
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

 private:
  class RendezvousData {
   public:
    explicit RendezvousData(size_t count)
        : mwait_(count), release_count_(0), payloads_(count) {}

    bool Release() { return release_count_.fetch_add(1) == 0; }

    void SetPayload(size_t ordinal, std::string payload) {
      std::lock_guard<std::mutex> lock(lock_);
      if (ordinal >= payloads_.size()) {
        status_ = ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                                 absl::StrCat("Invalid ordinal: ", ordinal));
      } else {
        payloads_[ordinal] = std::move(payload);
      }
    }

    ::grpc::Status Wait() {
      ::grpc::Status status =
          ToGrpcStatus(xla::util::CheckedCall([&]() { mwait_.Wait(); }));
      if (status.ok()) {
        std::lock_guard<std::mutex> lock(lock_);
        status = status_;
      }
      return status;
    }

    void Done() { mwait_.Done(); }

    const std::vector<std::string>& Payloads() const { return payloads_; };

   private:
    std::mutex lock_;
    util::MultiWait mwait_;
    std::atomic<size_t> release_count_;
    std::vector<std::string> payloads_;
    ::grpc::Status status_;
  };

  std::shared_ptr<RendezvousData> GetRendezvous(const std::string& tag) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = rendezvous_map_.find(tag);
    if (it == rendezvous_map_.end()) {
      it = rendezvous_map_
               .emplace(tag,
                        std::make_shared<RendezvousData>(config_.mesh_size()))
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
  response->mutable_config()->CopyFrom(config_);
  return ::grpc::Status::OK;
}

::grpc::Status MeshServiceImpl::Rendezvous(
    ::grpc::ServerContext* context, const grpc::RendezvousRequest* request,
    grpc::RendezvousResponse* response) {
  auto rendezvous = GetRendezvous(request->tag());
  rendezvous->SetPayload(request->ordinal(), request->payload());
  rendezvous->Done();
  TF_VLOG(3) << "Entering rendezvous: ordinal=" << request->ordinal()
             << " tag=" << request->tag() << " peer=" << context->peer();
  ::grpc::Status status = rendezvous->Wait();
  TF_VLOG(3) << "Exiting rendezvous: ordinal=" << request->ordinal()
             << " tag=" << request->tag() << " peer=" << context->peer()
             << " status=" << status;
  if (status.ok()) {
    for (auto& payload : rendezvous->Payloads()) {
      response->add_payloads(payload);
    }
  }
  ReleaseRendezvous(request->tag(), rendezvous);
  return status;
}

}  // namespace

struct MeshService::Impl {
  Impl(const std::string& address, grpc::Config config)
      : impl(std::move(config)) {
    ::grpc::ServerBuilder builder;
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
  grpc::GetConfigRequest reqeust;
  grpc::GetConfigResponse response;
  ::grpc::Status status = impl_->stub->GetConfig(&context, reqeust, &response);
  if (!status.ok()) {
    XLA_ERROR() << "Failed to retrieve mesh configuration: " << status;
  }
  return std::move(*response.mutable_config());
}

std::vector<std::string> MeshClient::Rendezvous(
    int ordinal, const std::string& tag, const std::string& payload) const {
  ::grpc::ClientContext context;
  grpc::RendezvousRequest reqeust;
  grpc::RendezvousResponse response;
  reqeust.set_tag(tag);
  reqeust.set_payload(payload);
  reqeust.set_ordinal(ordinal);
  TF_VLOG(3) << "Waiting for rendezvous: ordinal=" << ordinal << " tag=" << tag;
  ::grpc::Status status = impl_->stub->Rendezvous(&context, reqeust, &response);
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

}  // namespace service
}  // namespace xla
