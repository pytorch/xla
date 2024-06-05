#include "torch_xla/csrc/runtime/pjrt_compile_only.h"

namespace xla {

const Shape& CompileOnlyPjRtBuffer::on_device_shape() const { return shape_; }

PjRtMemorySpace* CompileOnlyPjRtBuffer::memory_space() const { return nullptr; }

PjRtDevice* CompileOnlyPjRtBuffer::device() const {
  // return device_.get();
  return nullptr;
}

PjRtClient* CompileOnlyPjRtBuffer::client() const {
  // return client_.get();
  return nullptr;
}

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

CompileOnlyPjRtDevice::CompileOnlyPjRtDevice(
    const PjRtDeviceDescription* description)
    : description_(std::move(description)) {}

const PjRtDeviceDescription& CompileOnlyPjRtDevice::description() const {
  return *description_;
}
void CompileOnlyPjRtDevice::SetClient(PjRtClient* client) { client_ = client; }

PjRtClient* CompileOnlyPjRtDevice::client() const { return client_; }
bool CompileOnlyPjRtDevice::IsAddressable() const { return false; }
int CompileOnlyPjRtDevice::local_hardware_id() const {
  return local_hardware_id_typed().value();
}

PjRtLocalDeviceId CompileOnlyPjRtDevice::local_device_id() const {
  return PjRtLocalDeviceId(local_hardware_id_typed().value());
}

PjRtLocalHardwareId CompileOnlyPjRtDevice::local_hardware_id_typed() const {
  return PjRtLocalHardwareId(-1);
}

std::unique_ptr<ScopedAsyncTrackingEvent>
CompileOnlyPjRtDevice::CreateAsyncTrackingEvent(
    absl::string_view description) const {
  return nullptr;
}
Status CompileOnlyPjRtDevice::TransferToInfeed(const LiteralSlice& literal) {
  return Unimplemented("TransferToInfeed is not supported");
}
Status CompileOnlyPjRtDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  return Unimplemented("TransferFromOutfeed is not supported");
}
absl::Span<PjRtMemorySpace* const> CompileOnlyPjRtDevice::memory_spaces()
    const {
  return {};
}
StatusOr<PjRtMemorySpace*> CompileOnlyPjRtDevice::default_memory_space() const {
  return Unimplemented("default_memory_space is not supported");
}

CompileOnlyPjRtClient::CompileOnlyPjRtClient(
    std::shared_ptr<PjRtTopologyDescription> topology)
    : topology_(std::move(topology)),
      descriptions_(topology_->DeviceDescriptions()) {
  // TODO(piz): figure out why the size should be 4 instead of topology_ size 8.
  descriptions_.resize(4);
  for (auto& description : descriptions_) {
    owned_devices_.push_back(
        std::make_unique<CompileOnlyPjRtDevice>(description.get()));
    owned_devices_.back()->SetClient(this);
    devices_.push_back(owned_devices_.back().get());
  }
}
int CompileOnlyPjRtClient::device_count() const { return devices().size(); }
int CompileOnlyPjRtClient::addressable_device_count() const { return 0; }
absl::Span<PjRtDevice* const> CompileOnlyPjRtClient::devices() const {
  return devices_;
}
absl::Span<PjRtDevice* const> CompileOnlyPjRtClient::addressable_devices()
    const {
  return devices_;
}
int CompileOnlyPjRtClient::process_index() const { return 0; }

StatusOr<PjRtDevice*> CompileOnlyPjRtClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  return Unimplemented("LookupDevice not available with compile-only client.");
}

StatusOr<PjRtDevice*> CompileOnlyPjRtClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  return Unimplemented(
      "LookupAddressableDevice not available with compile-only client.");
}
std::shared_ptr<PjRtTopologyDescription> CompileOnlyPjRtClient::get_topology()
    const {
  return topology_;
}

absl::Span<PjRtMemorySpace* const> CompileOnlyPjRtClient::memory_spaces()
    const {
  return {};
}

PjRtRuntimeType CompileOnlyPjRtClient::runtime_type() const {
  return PjRtRuntimeType::kTfrt;
}

absl::string_view CompileOnlyPjRtClient::platform_name() const { return "AOT"; }
absl::string_view CompileOnlyPjRtClient::platform_version() const {
  return topology_->platform_version();
}
PjRtPlatformId CompileOnlyPjRtClient::platform_id() const {
  return topology_->platform_id();
}
StatusOr<DeviceAssignment> CompileOnlyPjRtClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return Unimplemented(
      "GetDefaultDeviceAssignment not available with compile-only client.");
}
StatusOr<Layout> CompileOnlyPjRtClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return Unimplemented(
      "GetDefaultLayout not available with compile-only client.");
}

StatusOr<std::unique_ptr<HloCostAnalysis>>
CompileOnlyPjRtClient::GetHloCostAnalysis() const {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileOnlyPjRtClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileOnlyPjRtClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
CompileOnlyPjRtClient::DeserializeExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  return Unimplemented("DeserializeExecutable not implemented.");
}

StatusOr<std::unique_ptr<xla::PjRtBuffer>>
CompileOnlyPjRtClient::CreateUninitializedBuffer(const Shape& shape,
                                                 PjRtDevice* device) {
  return Unimplemented("CreateUninitializedBuffer not implemented.");
}

StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
CompileOnlyPjRtClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const Shape> shapes, PjRtMemorySpace* memory_space) {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
CompileOnlyPjRtClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const Shape> shapes, PjRtDevice* device) {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtBuffer>>
CompileOnlyPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtDevice* device) {
  // return Unimplemented{""};
  std::vector<Shape> tuple_shape;
  absl::Span<const bool> dynamic_dimensions;
  Shape shape(type, dims, dynamic_dimensions, tuple_shape);
  return std::make_unique<CompileOnlyPjRtBuffer>(
      shape, static_cast<CompileOnlyPjRtDevice*>(device), nullptr);
}

StatusOr<std::unique_ptr<PjRtBuffer>>
CompileOnlyPjRtClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                             PjRtDevice* device) {
  return Unimplemented("");
}

StatusOr<std::unique_ptr<PjRtBuffer>>
CompileOnlyPjRtClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream = std::nullopt) {
  return Unimplemented("");
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CompileOnlyPjRtClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* device,
    PjRtCrossHostRecvNotifier notifier) {
  return Unimplemented("");
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CompileOnlyPjRtClient::MakeCrossHostReceiveBuffersForGather(
    absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
    PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) {
  return Unimplemented("");
}

StatusOr<ChannelHandle> CompileOnlyPjRtClient::CreateChannelHandle() {
  return Unimplemented("");
}

StatusOr<ChannelHandle>
CompileOnlyPjRtClient::CreateDeviceToHostChannelHandle() {
  return Unimplemented("");
}

StatusOr<ChannelHandle>
CompileOnlyPjRtClient::CreateHostToDeviceChannelHandle() {
  return Unimplemented("");
}

Status CompileOnlyPjRtClient::Defragment() { return Unimplemented(""); }
}  // namespace xla