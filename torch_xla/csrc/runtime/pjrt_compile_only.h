#ifndef XLA_CLIENT_PJRT_COMPILE_ONLY_H_
#define XLA_CLIENT_PJRT_COMPILE_ONLY_H_

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"

namespace xla {

class CompileOnlyPjRtBuffer final : public PjRtBuffer {
 public:
  CompileOnlyPjRtBuffer(const Shape& shape) { shape_ = shape; }
  const Shape& on_device_shape() const override;
  PjRtMemorySpace* memory_space() const override;
  PjRtDevice* device() const override;
  PjRtClient* client() const override;
  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;
  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override;
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;
  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;
  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override;
  void Delete() override;
  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override;
  bool IsDeleted() override;
  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;
  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;
  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;
  void CopyToRemoteDeviceScattered(
      PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override;
  PjRtFuture<> GetReadyFuture() override;
  bool IsOnCpu() const override;

 private:
  Shape shape_;
};

}  // namespace xla

#endif
