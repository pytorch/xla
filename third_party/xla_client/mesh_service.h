#ifndef TENSORFLOW_COMPILER_XLA_RPC_XRT_MESH_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XRT_MESH_SERVICE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.pb.h"

namespace xla {
namespace service {

class MeshService {
  struct Impl;

 public:
  MeshService(const string& address, grpc::Config config);

  ~MeshService();

 private:
  std::unique_ptr<Impl> impl_;
};

class MeshClient {
  struct Impl;

 public:
  static MeshClient* Get();

  const string& address() const;

  grpc::Config GetConfig() const;

  void Rendezvous(const string& tag) const;

 private:
  MeshClient(const string& address);

  ~MeshClient();

  std::unique_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XRT_MESH_SERVICE_H_
