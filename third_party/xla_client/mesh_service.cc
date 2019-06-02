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
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.grpc.pb.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

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
  struct RendezvousData {
    explicit RendezvousData(size_t count) : mwait(count), release_count(0) {}

    util::MultiWait mwait;
    std::atomic<size_t> release_count;
  };

  std::shared_ptr<RendezvousData> GetRendezvous(const string& tag) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = rendezvous_map_.find(tag);
    if (it == rendezvous_map_.end()) {
      it =
          rendezvous_map_
              .emplace(tag,
                       std::make_shared<RendezvousData>(config_.workers_size()))
              .first;
    }
    return it->second;
  }

  void ReleaseRendezvous(const string& tag,
                         const std::shared_ptr<RendezvousData>& rendezvous) {
    if (rendezvous->release_count.fetch_add(1) == 0) {
      std::lock_guard<std::mutex> lock(lock_);
      rendezvous_map_.erase(tag);
    }
  }

  std::mutex lock_;
  grpc::Config config_;
  std::unordered_map<string, std::shared_ptr<RendezvousData>> rendezvous_map_;
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
  rendezvous->mwait.Done();
  TF_VLOG(3) << "Entering rendezvous: tag=" << request->tag()
             << " peer=" << context->peer();
  ::grpc::Status status = ToGrpcStatus(rendezvous->mwait.Wait());
  TF_VLOG(3) << "Exiting rendezvous: tag=" << request->tag()
             << " peer=" << context->peer() << " status=" << status;
  ReleaseRendezvous(request->tag(), rendezvous);
  return status;
}

}  // namespace

struct MeshService::Impl {
  Impl(const string& address, grpc::Config config) : impl(std::move(config)) {
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&impl);
    server = builder.BuildAndStart();
  }

  MeshServiceImpl impl;
  std::unique_ptr<::grpc::Server> server;
};

MeshService::MeshService(const string& address, grpc::Config config)
    : impl_(new Impl(address, std::move(config))) {}

MeshService::~MeshService() {}

struct MeshClient::Impl {
  explicit Impl(const string& address) : address(address) {
    std::shared_ptr<::grpc::Channel> channel =
        ::grpc::CreateChannel(address, ::grpc::InsecureChannelCredentials());
    stub = grpc::MeshService::NewStub(channel);
  }

  std::unique_ptr<grpc::MeshService::Stub> stub;
  string address;
};

MeshClient* MeshClient::Get() {
  auto create_client = []() {
    string mesh_service_address =
        sys_util::GetEnvString("XRT_MESH_SERVICE_ADDRESS", "");
    return !mesh_service_address.empty() ? new MeshClient(mesh_service_address)
                                         : nullptr;
  };
  static MeshClient* client = create_client();
  return client;
}

MeshClient::MeshClient(const string& address) : impl_(new Impl(address)) {}

MeshClient::~MeshClient() {}

const string& MeshClient::address() const { return impl_->address; }

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

void MeshClient::Rendezvous(const string& tag) const {
  ::grpc::ClientContext context;
  grpc::RendezvousRequest reqeust;
  grpc::RendezvousResponse response;
  reqeust.set_tag(tag);
  TF_VLOG(3) << "Waiting for rendezvous: " << tag;
  ::grpc::Status status = impl_->stub->Rendezvous(&context, reqeust, &response);
  TF_VLOG(3) << "Rendezvous wait complete: " << tag;
  if (!status.ok()) {
    XLA_ERROR() << "Failed to meet rendezvous '" << tag << "': " << status;
  }
}

}  // namespace service
}  // namespace xla
