#ifndef XLA_CLIENT_XRT_MESH_SERVICE_H_
#define XLA_CLIENT_XRT_MESH_SERVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/types.h"
#include "third_party/xla_client/mesh_service.pb.h"

namespace xla {
namespace service {

class MeshService {
  struct Impl;

 public:
  MeshService(const std::string& address, grpc::Config config);

  ~MeshService();

  void Shutdown();

 private:
  std::unique_ptr<Impl> impl_;
};

class MeshClient {
  struct Impl;

 public:
  static MeshClient* Get();

  const std::string& address() const;

  grpc::Config GetConfig(int ordinal) const;

  void SetConfig(int ordinal, const grpc::Config& config) const;

  std::vector<std::string> Rendezvous(int ordinal, const std::string& tag,
                                      const std::string& payload,
                                      absl::Span<const int64_t> replicas) const;

  std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas) const;

 private:
  MeshClient(const std::string& address);

  ~MeshClient();

  std::unique_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace xla

#endif  // XLA_CLIENT_XRT_MESH_SERVICE_H_
