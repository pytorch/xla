#include "torch_xla/csrc/runtime/distributed_runtime.h"

#include <stdlib.h>

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace runtime {

// Each instantiation and full specialization of std::atomic<> represents a type that different threads can simultaneously operate on (their instances), without raising undefined behavior.
std::atomic<xla::DistributedRuntimeClient*> distributed_runtime_client(nullptr);
std::once_flag distributed_runtime_client_once;

xla::DistributedRuntimeClient* CreateClient(int global_rank) {
  auto distributed_runtime = DistributedRuntime(global_rank);
  xla::DistributedRuntimeClient* client = distributed_runtime.GetClient().get();
  XLA_CHECK(client);
  return client;
}

xla::DistributedRuntimeClient* GetDistributedRuntimeClient(int global_rank) {
  std::call_once(distributed_runtime_client_once, 
  [&]() {distributed_runtime_client=std::move(CreateClient(global_rank)); });
  return distributed_runtime_client.load();
}




const std::string default_coordinator_port = "8547";

DistributedRuntime::DistributedRuntime(int global_rank) {
  std::string master_addr =
      runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
  std::string port = runtime::sys_util::GetEnvString("XLA_COORDINATOR_PORT",
                                                     default_coordinator_port);
  std::string dist_service_addr = absl::StrJoin({master_addr, port}, ":");
  if (global_rank == 0) {
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);
    XLA_CHECK(global_world_size > 0) << "WORLD_SIZE and LOCAL_WORLD_SIZE "
                                        "should not be empty at the same time.";
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = global_world_size;
    XLA_CHECK(
        xla::GetDistributedRuntimeService(dist_service_addr, service_options)
            .ok())
        << "Failed to initialize distributed runtime service.";
    dist_runtime_service_ =
        xla::GetDistributedRuntimeService(dist_service_addr, service_options)
            .value();
    //atexit(shutdown_service)
  }

  xla::DistributedRuntimeClient::Options client_options;
  client_options.node_id = global_rank;
  dist_runtime_client_ =
      xla::GetDistributedRuntimeClient(dist_service_addr, client_options);
  XLA_CHECK(dist_runtime_client_->Connect().ok())
      << "Failed to initialize distributed runtime client";
  //atexit(shutdown_client)
  std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": both dist runtime service/client are created." << std::endl;
}

DistributedRuntime::~DistributedRuntime() {
  std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": shutting down dist runtime client and service." << std::endl;
  if (dist_runtime_client_ != nullptr) {
    XLA_CHECK(dist_runtime_client_->Shutdown().ok())
        << "Failed to shut down the distributed runtime client.";
  }
  if (dist_runtime_service_ != nullptr) {
    dist_runtime_service_->Shutdown();
  }
}

std::shared_ptr<xla::DistributedRuntimeClient> DistributedRuntime::GetClient() {
  XLA_CHECK(dist_runtime_client_ != nullptr)
      << "distributed runtime client is null.";
  return dist_runtime_client_;
}

void DistributedRuntime::shutdown_service(void) {
  if (dist_runtime_service_ != nullptr) {
    dist_runtime_service_->Shutdown();
    dist_runtime_service_ = nullptr;
  }
}

void DistributedRuntime::shutdown_client(void) {
  if (dist_runtime_client_ != nullptr) {
    XLA_CHECK(dist_runtime_client_->Shutdown().ok())
        << "Failed to shut down the distributed runtime client.";
    dist_runtime_client_ = nullptr;
  }
}

}  // namespace runtime
}  // namespace torch_xla
