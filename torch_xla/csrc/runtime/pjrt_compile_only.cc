#include "torch_xla/csrc/runtime/pjrt_compile_only.h"

namespace xla {

const Shape& CompileOnlyPjRtBuffer::on_device_shape() const { return shape_; }

PjRtMemorySpace* CompileOnlyPjRtBuffer::memory_space() const { return nullptr; }

PjRtDevice* CompileOnlyPjRtBuffer::device() const { return nullptr; }

PjRtClient* CompileOnlyPjRtBuffer::client() const { return nullptr; }

StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
CompileOnlyPjRtBuffer::AcquireExternalReference() {
  return Unimplemented("");
}

PjRtFuture<> CompileOnlyPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  return PjRtFuture<>();
}

PjRtFuture<> CompileOnlyPjRtBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  return PjRtFuture<>();
}

StatusOr<size_t> CompileOnlyPjRtBuffer::GetOnDeviceSizeInBytes() const {
  return Unimplemented("");
}
PjRtFuture<> CompileOnlyPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                  int64_t transfer_size) {
  return PjRtFuture<>();
}

void CompileOnlyPjRtBuffer::Delete() { return; }

StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
CompileOnlyPjRtBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return Unimplemented("");
}

bool CompileOnlyPjRtBuffer::IsDeleted() { return false; }

StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  return Unimplemented("");
}
StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  return Unimplemented("");
}

void CompileOnlyPjRtBuffer::CopyToRemoteDevice(
    PjRtFuture<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  return;
}

void CompileOnlyPjRtBuffer::CopyToRemoteDeviceScattered(
    PjRtFuture<std::vector<std::string>> serialized_descriptors,
    std::vector<RemoteSendCallback> callbacks,
    const ScatterDetails& scatter_details) {
  return;
}

PjRtFuture<> CompileOnlyPjRtBuffer::GetReadyFuture() { return PjRtFuture<>(); }

bool CompileOnlyPjRtBuffer::IsOnCpu() const { return false; }

}  // namespace xla
