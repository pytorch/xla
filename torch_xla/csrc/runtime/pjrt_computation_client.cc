#include "torch_xla/csrc/runtime/pjrt_computation_client.h"

#include <algorithm>
#include <future>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_hash.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/operation_manager.h"
#include "torch_xla/csrc/runtime/pjrt_registry.h"
#include "torch_xla/csrc/runtime/profiler.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/tensor_source.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "torch_xla/csrc/thread_pool.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/protobuf_util.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"

using xla::internal::XlaBuilderFriend;

namespace torch_xla {
namespace runtime {

namespace {

// Builds a map from the device's global ordinal to its index in the `devices`
// array.
std::unordered_map<int, int> build_index_map(
    const std::vector<std::string>& devices) {
  std::unordered_map<int, int> device_index;
  for (int i = 0; i < devices.size(); ++i) {
    std::vector<std::string> device_spec = absl::StrSplit(devices[i], ':');
    XLA_CHECK_EQ(device_spec.size(), 2)
        << "Invalid device specification: " << devices[i];
    int global_ordinal = std::stoi(device_spec[1]);
    device_index[global_ordinal] = i;
  }
  return device_index;
}

// Builds the xla::Shape of the output xla::Literal on the host.
xla::Shape host_output_shape(xla::PjRtBuffer* buffer) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      buffer->element_type(), buffer->logical_dimensions().value());
  *shape.mutable_layout() = xla::GetXlaLayoutUnsafe(buffer->layout());

  return xla::ShapeUtil::DeviceShapeToHostShape(shape);
}

torch::lazy::hash_t hash_comp_env(
    xla::PjRtClient* client, std::vector<xla::PjRtDevice*>& ordered_devices) {
  torch::lazy::hash_t hash = hash::HashXlaEnvVars();
  auto topology_desc = client->GetTopologyDescription();
  if (topology_desc.ok()) {
    // Some backends support a topology description which provides a better
    // view of the specific compilation environment.
    auto serialized = topology_desc.value()->Serialize();
    if (serialized.ok()) {
      return torch::lazy::HashCombine(
          hash,
          torch::lazy::DataHash(serialized->data(), serialized->length()));
    }
    // If serialization fails, fallthrough to the manual approach.
  }
  std::string platform_name(client->platform_name());
  std::string platform_version(client->platform_version());
  hash = torch::lazy::HashCombine(
      hash, torch::lazy::StringHash(platform_name.c_str()));
  // platform_version incorporates libtpu version and hardware type.
  hash = torch::lazy::HashCombine(
      hash, torch::lazy::StringHash(platform_version.c_str()));
  // Include global devices in the hash, ensuring order is consistent.
  for (auto& device : ordered_devices) {
    std::string device_str(device->ToString());
    hash = torch::lazy::HashCombine(
        hash, torch::lazy::StringHash(device_str.c_str()));
  }
  return hash;
}

}  // namespace

std::string PjRtComputationClient::PjRtDeviceToString(
    xla::PjRtDevice* const device) const {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  int ordinal = global_ordinals_.at(device->id());
  std::string str = absl::StrFormat("%s:%d", platform, ordinal);
  return str;
}

std::vector<std::string> PjRtComputationClient::PjRtDevicesToString(
    absl::Span<xla::PjRtDevice* const> devices) const {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(PjRtDeviceToString(device));
  }

  return strs;
}

PjRtComputationClient::PjRtComputationClient() {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");
  std::tie(client_, coordinator_) = InitializePjRt(device_type);

  // PjRtDevice IDs are not guaranteed to be dense, so we need to track
  // a device's global ordinal separately from its device ID. Order the
  // devices by increasing ID to assign global ordinals.
  std::vector<xla::PjRtDevice*> ordered_devices(client_->device_count());
  std::partial_sort_copy(client_->devices().begin(), client_->devices().end(),
                         ordered_devices.begin(), ordered_devices.end(),
                         [](auto& a, auto& b) { return a->id() < b->id(); });
  for (auto* device : ordered_devices) {
    global_ordinals_[device->id()] = global_ordinals_.size();
    std::string device_str = PjRtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
  }
  comp_env_hash_ = hash_comp_env(client_.get(), ordered_devices);

  auto tracked_devices = GetLocalDevices();
  tracked_devices.emplace_back(spmd_device_str);
  operation_manager_ = std::move(OperationManager(std::move(tracked_devices)));
}

PjRtComputationClient::~PjRtComputationClient() {
  // In the GPU case, the PjRtClient depends on the DistributedRuntimeClient
  // tracked in XlaCoordinator, so the PjRtClient must be destroyed first.
  client_ = nullptr;
  coordinator_ = nullptr;
}

bool PjRtComputationClient::CoordinatorInitialized() const {
  return coordinator_ != nullptr;
}

void PjRtComputationClient::InitializeCoordinator(int global_rank,
                                                  int world_size,
                                                  std::string master_addr,
                                                  std::string port) {
  XLA_CHECK(coordinator_ == nullptr)
      << "Can only initialize the XlaCoordinator once.";
  coordinator_ = std::make_unique<XlaCoordinator>(global_rank, world_size,
                                                  master_addr, port);
}

XlaCoordinator& PjRtComputationClient::GetCoordinator() {
  XLA_CHECK(coordinator_ != nullptr)
      << "XlaCoordinator has not been initialized";
  return *coordinator_;
}

void PjRtComputationClient::PjRtData::Assign(
    const torch::lazy::BackendData& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtComputationClient::CreateDataPlaceholder(
    std::string device, xla::Shape shape,
    std::optional<xla::OpSharding> sharding) {
  if (sharding.has_value()) {
    return std::make_shared<PjRtShardedData>(
        std::move(device), std::move(shape), std::move(*sharding));
  }

  return std::make_shared<PjRtData>(std::move(device), std::move(shape));
}

ComputationClient::DataPtr PjRtComputationClient::CreateData(
    std::string device, xla::Shape shape,
    std::shared_ptr<xla::PjRtBuffer> pjrt_buffer) {
  return std::make_shared<PjRtData>(std::move(device), std::move(shape),
                                    pjrt_buffer);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::GetDataShards(
    ComputationClient::DataPtr data) {
  tsl::profiler::TraceMe activity("PjRtComputationClient::GetDataShards",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> shards;
  if (PjRtShardedData* sharded_data =
          dynamic_cast<PjRtShardedData*>(data.get())) {
    for (auto shard : sharded_data->shards) {
      shards.push_back(std::make_shared<PjRtData>(
          shard->device(), shard->shape(), shard->buffer));
    }
  } else {
    shards.push_back(data);
  }
  return shards;
}

ComputationClient::DataPtr PjRtComputationClient::GetDataShard(
    ComputationClient::DataPtr data, size_t index) {
  tsl::profiler::TraceMe activity("PjRtComputationClient::GetDataShard",
                                  tsl::profiler::TraceMeLevel::kInfo);
  if (PjRtShardedData* sharded_data =
          dynamic_cast<PjRtShardedData*>(data.get())) {
    XLA_CHECK_LE(index, sharded_data->shards.size())
        << "GetDataShard out of range with index: " << index
        << " and num of shard: " << sharded_data->shards.size();
    std::shared_ptr<PjRtData> shard = sharded_data->shards[index];
    return std::make_shared<PjRtData>(shard->device(), shard->shape(),
                                      shard->buffer);
  } else {
    return data;
  }
}

ComputationClient::DataPtr PjRtComputationClient::WrapDataShards(
    absl::Span<const DataPtr> shards, std::string device, xla::Shape shape,
    xla::OpSharding sharding) {
  XLA_CHECK_EQ(shards.size(), client_->addressable_devices().size());
  std::vector<std::shared_ptr<PjRtData>> pjrt_data_shards;
  pjrt_data_shards.reserve(shards.size());
  for (auto& shard : shards) {
    XLA_CHECK(shard != nullptr);
    auto pjrt_shard = dynamic_cast<PjRtData*>(shard.get());
    pjrt_data_shards.push_back(std::make_shared<PjRtData>(
        pjrt_shard->device(), pjrt_shard->shape(), pjrt_shard->buffer));
  }
  return std::make_shared<PjRtShardedData>(device, shape, pjrt_data_shards,
                                           sharding);
}

std::optional<xla::OpSharding> PjRtComputationClient::GetDataSharding(
    DataPtr handle) {
  if (auto sharded_data = dynamic_cast<PjRtShardedData*>(handle.get())) {
    return sharded_data->GetSharding();
  }
  return std::optional<xla::OpSharding>();
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  metrics::TimedSection timed(TransferToDeviceMetric());
  tsl::profiler::TraceMe activity("PjRtComputationClient::TransferToDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  int64_t total_size = 0;
  for (auto& tensor : tensors) {
    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(tensor->device());

    total_size += xla::ShapeUtil::ByteSizeOf(tensor->shape());

    std::shared_ptr<xla::PjRtBuffer> buffer =
        std::move(client_
                      ->BufferFromHostBuffer(
                          tensor->data(), tensor->primitive_type(),
                          tensor->dimensions(), tensor->byte_strides(),
                          xla::PjRtClient::HostBufferSemantics::
                              kImmutableUntilTransferCompletes,
                          [tensor]() { /* frees tensor */ }, pjrt_device)
                      .value());

    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensor->device(), tensor->shape(), buffer);
    datas.push_back(data);
  }
  OutboundDataMetric()->AddSample(total_size);
  CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr PjRtComputationClient::TransferShardsToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
    std::string device, xla::Shape shape, xla::OpSharding sharding) {
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::TransferShardsToDevice",
      tsl::profiler::TraceMeLevel::kInfo);
  // TODO(jonbolin): Consider using CopyToDevice when sharding is REPLICATED.
  // We are opting out of CopyToDevice for now due to the synchronization
  // issues observed in ShardingUtil::InputHandler, but because CopyToDevice
  // directly copies buffers between devices using ICI, it can be much faster
  // than transferring from the host to each device.
  auto data_shards = TransferToDevice(tensor_shards);
  std::vector<std::shared_ptr<PjRtData>> pjrt_data_shards;
  for (auto& shard : data_shards) {
    auto pjrt_shard = dynamic_cast<PjRtData*>(shard.get());
    pjrt_data_shards.push_back(std::make_shared<PjRtData>(
        pjrt_shard->device(), pjrt_shard->shape(), pjrt_shard->buffer));
  }
  return std::make_shared<PjRtShardedData>(device, shape, pjrt_data_shards,
                                           sharding);
}

ComputationClient::DataPtr PjRtComputationClient::CopyToDevice(
    ComputationClient::DataPtr data, std::string dst) {
  tsl::profiler::TraceMe activity("PjRtComputationClient::CopyToDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(data.get());
  XLA_CHECK(pjrt_data->HasValue()) << "Can't copy invalid device data.";

  xla::PjRtDevice* dst_device = StringToPjRtDevice(dst);
  XLA_CHECK(dst_device->IsAddressable()) << dst << "is not addressable.";

  // Returns error if the buffer is already on `dst_device`.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> status_or =
      pjrt_data->buffer->CopyToDevice(dst_device);
  if (!status_or.ok()) {
    return data;
  }
  return std::make_shared<PjRtData>(dst, pjrt_data->shape(),
                                    std::move(status_or.value()));
}

std::shared_ptr<PjRtComputationClient::PjRtData>
PjRtComputationClient::ReplicateShardedData(
    const ComputationClient::DataPtr& handle) {
  if (auto unsharded_data = std::dynamic_pointer_cast<PjRtData>(handle)) {
    return unsharded_data;
  } else if (auto sharded_data =
                 std::dynamic_pointer_cast<PjRtShardedData>(handle)) {
    XLA_COUNTER("ReplicateShardedData", 1);
    TF_VLOG(1) << "ReplicateShardedData (handle=" << sharded_data->GetHandle()
               << ", shape=" << sharded_data->shape() << ")";
    if (sharded_data->GetSharding().type() == xla::OpSharding::REPLICATED) {
      // Data is replicated, return the first shard
      return sharded_data->shards[0];
    }
    xla::XlaBuilder builder("ReplicateShardedData");
    xla::Shape shape = sharded_data->shape();
    builder.SetSharding(sharded_data->GetSharding());

    // perform a simple identity calculation to reassemble the input as
    // replicated output.
    xla::XlaOp x = xla::Parameter(&builder, 0, shape, "p0");
    builder.SetSharding(xla::HloSharding::Replicate().ToProto());
    xla::XlaOp scalar_zero_op = xla::ConvertElementType(
        xla::ConstantR0(&builder, 0), shape.element_type());
    xla::XlaOp y = xla::Add(x, scalar_zero_op);
    auto instruction = XlaBuilderFriend::GetInstruction(y);
    *instruction->mutable_sharding() = xla::HloSharding::Replicate().ToProto();

    xla::XlaComputation computation =
        ConsumeValue(builder.Build(/*remove_dynamic_dimensions=*/false));
    xla::ProgramShape program_shape =
        ConsumeValue(computation.GetProgramShape());

    std::string device = GetDefaultDevice();
    std::vector<torch_xla::runtime::ComputationClient::CompileInstance>
        instances;
    instances.push_back({std::move(computation), device,
                         GetCompilationDevices(device, {}), &shape,
                         /*should_wrap_parameter=*/false,
                         /*is_sharded=*/true,
                         /*allow_spmd_sharding_propagation_to_output=*/false});
    std::vector<
        std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>>
        computations = Compile(std::move(instances));

    torch_xla::runtime::ComputationClient::ExecuteReplicatedOptions
        execute_options;
    auto sharded_results =
        ExecuteReplicated(*computations.front(), {sharded_data},
                          GetLocalDevices(), execute_options);
    XLA_CHECK(sharded_results.size() > 0)
        << "empty ExecuteReplicated results returned.";
    XLA_CHECK(sharded_results.size() == 1)
        << "Wrong number of outputs, expected: 1, actual: "
        << sharded_results.size();
    return std::dynamic_pointer_cast<PjRtShardedData>(sharded_results[0])
        ->shards[0];
  }

  XLA_ERROR() << "Data must be PjRtData or PjRtShardedData, got "
              << handle->ToString();
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::ReshardData(
    absl::Span<const ComputationClient::DataPtr> handles,
    absl::Span<const xla::OpSharding> shardings) {
  tsl::profiler::TraceMe activity("ReshardData",
                                  tsl::profiler::TraceMeLevel::kInfo);
  XLA_COUNTER("ReshardData", 1);
  XLA_CHECK_EQ(handles.size(), shardings.size())
      << "input handles and shardings must have the same length.";
  XLA_CHECK(UseVirtualDevice()) << "We only supports SPMD mode resharding.";

  // Perform a simple identity calculation to reshard.
  xla::XlaBuilder builder("ReshardData");

  std::vector<xla::Shape> shapes;
  shapes.reserve(handles.size());
  std::vector<xla::HloSharding> hlo_shardings;
  hlo_shardings.reserve(handles.size());
  std::vector<xla::XlaOp> param_ops;
  param_ops.reserve(handles.size());
  for (int i = 0; i < handles.size(); ++i) {
    PjRtShardedData* sharded_data =
        dynamic_cast<PjRtShardedData*>(handles[i].get());
    XLA_CHECK_NE(sharded_data, nullptr)
        << "Resharding requires PjRtShardedData on SPMD virtual device, "
        << "current device: " << handles[i]->device();
    shapes.push_back(sharded_data->shape());

    const xla::OpSharding& sharding = shardings[i];
    XLA_CHECK_NE(sharding.type(), xla::OpSharding::UNKNOWN)
        << "Resharding by UNKNOWN sharding type is not allowed.";

    hlo_shardings.push_back(
        ConsumeValue(xla::HloSharding::FromProto(sharding)));

    xla::OpSharding fallback_sharding;
    fallback_sharding.set_type(xla::OpSharding::REPLICATED);
    xla::XlaScopedShardingAssignment assign(
        &builder, sharded_data->GetSharding().type() == xla::OpSharding::UNKNOWN
                      ? fallback_sharding
                      : sharded_data->GetSharding());
    param_ops.push_back(
        xla::Parameter(&builder, i, shapes[i], absl::StrCat("p.", i)));
  }

  xla::XlaOp root;
  {
    xla::Shape shapes_tuple = xla::ShapeUtil::MakeTupleShape(shapes);
    XLA_CHECK_EQ(shapes_tuple.tuple_shapes_size(), hlo_shardings.size());
    xla::HloSharding new_shardings_tuple =
        xla::HloSharding::Tuple(shapes_tuple, hlo_shardings);
    xla::XlaScopedShardingAssignment assign(&builder,
                                            new_shardings_tuple.ToProto());
    root = xla::Tuple(&builder, param_ops);
  }

  xla::XlaComputation xla_computation = ConsumeValue(builder.Build(root));
  xla::ProgramShape program_shape =
      ConsumeValue(xla_computation.GetProgramShape());

  std::string device = GetDefaultDevice();
  std::vector<torch_xla::runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(xla_computation), device,
                       GetCompilationDevices(device, {}),
                       &program_shape.result(),
                       /*should_wrap_parameter=*/false,
                       /*is_sharded=*/true,
                       /*allow_spmd_sharding_propagation_to_output=*/false});
  std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>
      computation = Compile(std::move(instances)).front();

  torch_xla::runtime::ComputationClient::ExecuteReplicatedOptions
      execute_options;
  auto resharded_results = ExecuteReplicated(
      *computation, handles, GetLocalDevices(), execute_options);
  return resharded_results;
}

std::uintptr_t PjRtComputationClient::UnsafeBufferPointer(
    const DataPtr handle) {
  std::shared_ptr<PjRtData> pjrt_data =
      std::dynamic_pointer_cast<PjRtData>(handle);
  XLA_CHECK(pjrt_data) << "handle must be PjRtData, got " << handle->ToString();
  XLA_CHECK(pjrt_data->buffer != nullptr)
      << "PjRt buffer is null in " << __FUNCTION__;
  absl::StatusOr<std::uintptr_t> ptr =
      client_->UnsafeBufferPointer(pjrt_data->buffer.get());
  XLA_CHECK(ptr.ok());
  return ptr.value();
}

std::shared_ptr<xla::PjRtBuffer> PjRtComputationClient::GetPjRtBuffer(
    const DataPtr handle) {
  std::shared_ptr<PjRtData> pjrt_data =
      std::dynamic_pointer_cast<PjRtData>(handle);

  XLA_CHECK(pjrt_data) << "handle must be PjRtData, got " << handle->ToString();
  std::shared_ptr<xla::PjRtBuffer> pjrt_buffer = pjrt_data->buffer;
  if (pjrt_buffer != nullptr) {
    return pjrt_buffer;
  } else {
    TF_VLOG(3) << "The pjrt buffer is null so we need to wait for device ops "
                  "to finish.";
    WaitDeviceOps({});
    return std::dynamic_pointer_cast<PjRtData>(handle)->buffer;
  }
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromDevice(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromDeviceMetric());
  tsl::profiler::TraceMe activity("PjRtComputationClient::TransferFromDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<xla::PjRtFuture<>> futures;
  futures.reserve(handles.size());
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  int64_t total_size = 0;
  for (auto handle : handles) {
    // Use XLA replication to reassemble the sharded data. If input handle
    // is not sharded, then it is a no-op.
    std::shared_ptr<PjRtData> pjrt_data = ReplicateShardedData(handle);
    XLA_CHECK(pjrt_data) << "PjRt_data is null in " << __FUNCTION__;
    XLA_CHECK(pjrt_data->buffer != nullptr)
        << "PjRt buffer is null in " << __FUNCTION__;

    xla::Literal& literal =
        literals.emplace_back(host_output_shape(pjrt_data->buffer.get()));
    futures.push_back(pjrt_data->buffer->ToLiteral(&literal));

    total_size += literal.size_bytes();
  }
  for (auto& future : futures) {
    absl::Status status = future.Await();
    XLA_CHECK_OK(status) << "Failed to await future from buffer to literal in"
                         << __FUNCTION__;
  }
  InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  auto metrics_fn = CompileMetric;
  if (instances[0].eager_mode) {
    metrics_fn = EagerCompileMetric;
  }
  metrics::TimedSection timed(metrics_fn());
  tsl::profiler::TraceMe activity("PjRtComputationClient::Compile",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::ComputationPtr> computations;

  for (auto& instance : instances) {
    xla::CompileOptions compile_options;
    if (instance.is_sharded) {
      // TODO(yeounoh) multi-host, multi-slice configurations
      compile_options.executable_build_options.set_use_spmd_partitioning(true);

      // We can override the compiler's default behavior to replicate the
      // outputs. Setting this to true would wrapping the sharded outputs in
      // PjRtShardedData.
      compile_options.executable_build_options
          .set_allow_spmd_sharding_propagation_to_output(
              {instance.allow_spmd_sharding_propagation_to_output});

      int num_partitions = client_->device_count();
      compile_options.executable_build_options.set_num_partitions(
          num_partitions);
      compile_options.executable_build_options.set_num_replicas(1);
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;
      compile_options.executable_build_options.set_use_auto_spmd_partitioning(
          instance.use_auto_spmd_partitioning);
      TF_VLOG(3) << "Auto SPMD partitioning "
                 << (instance.use_auto_spmd_partitioning ? "enabled!"
                                                         : "disabled.");
      if (!instance.auto_spmd_mesh_shape.empty()) {
        compile_options.executable_build_options
            .set_auto_spmd_partitioning_mesh_shape(
                instance.auto_spmd_mesh_shape);
        TF_VLOG(3) << "auto_spmd_partitioning_mesh_shape="
                   << absl::StrJoin(compile_options.executable_build_options
                                        .auto_spmd_partitioning_mesh_shape(),
                                    ",");
      }
      if (!instance.auto_spmd_mesh_ids.empty()) {
        compile_options.executable_build_options
            .set_auto_spmd_partitioning_mesh_ids(instance.auto_spmd_mesh_ids);
        TF_VLOG(3) << "auto_spmd_partitioning_mesh_ids="
                   << absl::StrJoin(compile_options.executable_build_options
                                        .auto_spmd_partitioning_mesh_ids(),
                                    ",");
      }

      // TODO(244391366) verify this is correct for the collectives ops
      xla::DeviceAssignment device_assignment(1, client_->device_count());
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto& [device_id, global_ordinal] : global_ordinals_) {
        device_assignment(0, global_ordinal) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    } else {
      // TODO(wcromar): set compile_options.argument_layouts, enable strict
      // shapes
      compile_options.executable_build_options.set_num_partitions(1);
      compile_options.executable_build_options.set_num_replicas(
          client_->device_count());
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

      xla::DeviceAssignment device_assignment(client_->device_count(), 1);
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto& [device_id, global_ordinal] : global_ordinals_) {
        device_assignment(global_ordinal, 0) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    }

    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
    if (runtime::sys_util::GetEnvBool("XLA_STABLEHLO_COMPILE", false)) {
      // Convert HLO to StableHLO for PjRt client compilation.
      mlir::MLIRContext context;
      mlir::ModuleOp mlir_module =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      ConvertHloToStableHlo(instance.computation.mutable_proto(), &mlir_module);
      executable = client_->Compile(mlir_module, compile_options).value();
      StableHloCompileCounter()->AddValue(1);
    } else {
      executable =
          client_->Compile(instance.computation, compile_options).value();
    }

    auto memory_stats_status_or = executable->GetCompiledMemoryStats();
    if (memory_stats_status_or.ok()) {
      xla::CompiledMemoryStats memory_stats = memory_stats_status_or.value();
      TF_VLOG(3) << "memory usage detail = " << memory_stats.DebugString();
    } else {
      TF_VLOG(3) << "memory usage is not availiable";
    }

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    xla::HloComputation* hlo_computation = hlo_modules[0]->entry_computation();
    std::shared_ptr<PjRtComputation> pjrt_computation =
        std::make_shared<PjRtComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            instance.devices, std::move(executable));

    computations.push_back(pjrt_computation);

    CreateCompileHandlesCounter()->AddValue(1);
  }

  return computations;
}

std::string PjRtComputationClient::SerializeComputation(
    const ComputationPtr computation) {
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(*computation);

  return ConsumeValue(pjrt_computation.executable->SerializeExecutable());
}

ComputationClient::ComputationPtr PjRtComputationClient::DeserializeComputation(
    const std::string& serialized) {
  auto executable_or = client_->DeserializeExecutable(serialized, std::nullopt);
  if (!executable_or.ok()) {
    TF_LOG(WARNING) << "Failed to deserialize executable: "
                    << executable_or.status();
    return nullptr;
  }
  auto executable = std::move(*executable_or);

  auto hlo_modules = executable->GetHloModules();
  if (!hlo_modules.ok()) {
    TF_LOG(WARNING)
        << "Failed to retrieve HLO modules from deserialized executable";
    return nullptr;
  }
  XLA_CHECK(hlo_modules->size() == 1)
      << "Only a single module is supported for persistent computation "
         "caching. Please unset the XLA_PERSISTENT_CACHE_PATH "
         "variable to disable persistent caching.";
  xla::XlaComputation computation((*hlo_modules)[0]->ToProto());

  std::vector<std::string> devices = {UseVirtualDevice() ? spmd_device_str
                                                         : GetDefaultDevice()};
  return std::make_shared<PjRtComputation>(std::move(computation), devices,
                                           std::move(executable));
}

torch::lazy::hash_t PjRtComputationClient::HashCompilationEnv() {
  // TODO(jonbolin): Incorporate CompileOptions into the hash. These are
  // deterministically generated at the moment, so they don't need to be
  // included. It will require a small refactor, so punting on this for now.
  return comp_env_hash_;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteComputation` and the async work in `ExecuteSharded` are
  // complete; a copy is held from the lambda that releases it when done.
  auto metrics_fn = ExecuteMetric;
  if (options.eager_mode) {
    metrics_fn = EagerExecuteMetric;
  }
  auto timed = std::make_shared<metrics::TimedSection>(metrics_fn());
  tsl::profiler::TraceMe activity("PjRtComputationClient::ExecuteComputation",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TF_VLOG(1) << "Executing PjRt computation on " << device;
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);

  xla::PjRtDevice* pjrt_device = StringToPjRtDevice(device);
  XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
    const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

    XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
        << "The device currently being used : " << pjrt_device->DebugString()
        << " is different from the device where the buffer resides: "
        << pjrt_data->buffer->device()->DebugString();
    buffers.push_back(pjrt_data->buffer.get());
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = false;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;

  TF_VLOG(5) << "ExecuteComputation acquiring PJRT device lock for " << device;
  auto op_tracker = operation_manager_.StartOperation(device);
  TF_VLOG(5) << "ExecuteComputation acquiring PJRT device lock for " << device
             << " Done";

  std::optional<xla::PjRtFuture<>> returned_future;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
      pjrt_computation.executable
          ->ExecuteSharded(buffers, pjrt_device, execute_options,
                           returned_future)
          .value();

  returned_future->OnReady(std::move(
      [timed, op_tracker = std::move(op_tracker)](absl::Status unused) mutable {
        timed.reset();
        TF_VLOG(3) << "ExecuteComputation returned_future->OnReady finished";
      }));

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

    std::shared_ptr<PjRtData> data =
        std::make_shared<PjRtData>(device, std::move(buffer));

    datas.push_back(data);
  }
  CreateDataHandlesCounter()->AddValue(datas.size());

  TF_VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteReplicated` and the async work in `Execute` are
  // complete; a copy is held from the lambda that releases it when done.
  auto timed =
      std::make_shared<metrics::TimedSection>(ExecuteReplicatedMetric());
  tsl::profiler::TraceMe activity("PjRtComputationClient::ExecuteReplicated",
                                  tsl::profiler::TraceMeLevel::kInfo);
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);

  std::vector<std::vector<xla::PjRtBuffer*>> argument_handles(
      devices.size(), std::vector<xla::PjRtBuffer*>(arguments.size()));
  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_argument_handle",
        tsl::profiler::TraceMeLevel::kInfo);

    absl::BlockingCounter counter(arguments.size());

    // Time in nanoseconds that it takes to prepare an argument. Used to tune
    // number of threads spawned by ParallelFor. Measured on 2023/11/28.
    static constexpr int64_t argument_handle_cost_ns = 10000;
    pool_.ParallelFor(
        arguments.size(), argument_handle_cost_ns,
        [&](int64_t start, int64_t end) {
          for (int32_t i = start; i < end; ++i) {
            auto pjrt_data =
                std::dynamic_pointer_cast<PjRtShardedData>(arguments[i]);
            XLA_CHECK_EQ(pjrt_data->shards.size(), devices.size())
                << "Expected one shard per device";

            for (int32_t d = 0; d < devices.size(); d++) {
              std::shared_ptr<PjRtData> shard = pjrt_data->shards[d];

              xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[d]);
              XLA_CHECK_EQ(shard->buffer->device(), pjrt_device);
              XLA_CHECK(pjrt_device->IsAddressable())
                  << pjrt_device->DebugString();

              argument_handles[d][i] = shard->buffer.get();
            }
            counter.DecrementCount();
          };
        });
    counter.Wait();
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = true;
  // TODO(yeounoh) currently only support single-slice execution
  execute_options.multi_slice_config = nullptr;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;

  // Grab the shared lock and block the `WaitDeviceOps` until buffer is
  // ready. Since this is the SPMD code path. There is no points to grab
  // devices lock for every individual device.
  TF_VLOG(5) << "ExecuteReplicated acquiring PJRT device lock for "
             << spmd_device_str;
  auto op_tracker = operation_manager_.StartOperation(spmd_device_str);
  TF_VLOG(5) << "ExecuteReplicated acquiring PJRT device lock for "
             << spmd_device_str << " Done";

  std::optional<std::vector<xla::PjRtFuture<>>> returned_futures =
      std::vector<xla::PjRtFuture<>>();
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results;
  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_execute",
        tsl::profiler::TraceMeLevel::kInfo);
    results = pjrt_computation.executable
                  ->Execute(std::move(argument_handles), execute_options,
                            returned_futures)
                  .value();

    (*returned_futures)[0].OnReady(
        std::move([timed, op_tracker = std::move(op_tracker)](
                      absl::Status unused) mutable {
          timed.reset();
          TF_VLOG(3) << "ExecuteReplicated returned_future->OnReady finished";
        }));
  }

  size_t num_outputs = results[0].size();
  std::vector<ComputationClient::DataPtr> data_handles(num_outputs);

  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_result_handle",
        tsl::profiler::TraceMeLevel::kInfo);

    const xla::Shape& result_shape = computation.program_shape().result();
    TF_VLOG(3) << "Processing output with shape " << result_shape.ToString();
    const std::vector<xla::Shape>& output_shapes =
        result_shape.IsTuple() ? result_shape.tuple_shapes()
                               : std::vector<xla::Shape>({result_shape});
    XLA_CHECK_EQ(output_shapes.size(), num_outputs);

    const std::vector<xla::OpSharding>& output_shardings =
        pjrt_computation.output_shardings_.has_value() && num_outputs > 0
            ? *pjrt_computation.output_shardings_
            :
            // Without an explicit sharding annotation, the output is implicitly
            // replicated, and we mark explicitly replicated here.
            std::vector<xla::OpSharding>(num_outputs);
    XLA_CHECK_EQ(output_shardings.size(), num_outputs);

    absl::BlockingCounter counter(num_outputs);

    // Time in nanoseconds that it takes to process a result buffer.
    // Measured on 2023/11/28.
    static constexpr int64_t result_handle_cost_ns = 10000;
    pool_.ParallelFor(
        num_outputs, result_handle_cost_ns, [&](int64_t start, int64_t end) {
          for (int32_t i = start; i < end; ++i) {
            std::vector<std::shared_ptr<PjRtData>> shards(devices.size());
            for (int32_t d = 0; d < devices.size(); d++) {
              std::unique_ptr<xla::PjRtBuffer> buffer =
                  std::move(results[d][i]);
              shards[d] =
                  std::make_shared<PjRtData>(devices[d], std::move(buffer));
            }

            data_handles[i] = std::make_shared<PjRtShardedData>(
                spmd_device_str, output_shapes[i], std::move(shards),
                output_shardings[i]);
            TF_VLOG(5) << "Created sharded data with shape "
                       << data_handles[i]->shape().ToString();
            counter.DecrementCount();
          }
        });
    counter.Wait();
  }

  TF_VLOG(1) << "Returning " << data_handles.size() << " sharded outputs.";
  return data_handles;
}

size_t PjRtComputationClient::GetNumDevices() const {
  return client_->addressable_device_count();
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return PjRtDeviceToString(client_->addressable_devices()[0]);
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  return PjRtDevicesToString(client_->addressable_devices());
}

std::vector<std::string> PjRtComputationClient::GetAllDevices() const {
  return PjRtDevicesToString(client_->devices());
}

int PjRtComputationClient::GetNumProcesses() const {
  int max_process_index = client_->process_index();
  for (auto* device : client_->devices()) {
    max_process_index = std::max(max_process_index, device->process_index());
  }

  return max_process_index + 1;
};

const absl::flat_hash_map<
    std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>
PjRtComputationClient::GetDeviceAttributes(const std::string& device) {
  return PjRtComputationClient::StringToPjRtDevice(device)->Attributes();
}

void PjRtComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
PjRtComputationClient::GetReplicationDevices() {
  return replication_devices_;
}

xla::PjRtDevice* PjRtComputationClient::StringToPjRtDevice(
    const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
      << "Unknown device " << device;
  xla::PjRtDevice* pjrt_device = string_to_device_[device];
  return pjrt_device;
}

void PjRtComputationClient::WaitDeviceOps(
    absl::Span<const std::string> devices) {
  TF_VLOG(3) << "Waiting for " << absl::StrJoin(devices, ", ");
  operation_manager_.WaitForDevices(
      devices.empty()
          ? (UseVirtualDevice() ? std::vector<std::string>({spmd_device_str})
                                : GetLocalDevices())
          : devices);
}

std::map<std::string, Metric> PjRtComputationClient::GetMetrics() const {
  // TODO(jonbolin): Add any PJRt-client-specific metrics here
  return {};
}

ComputationClient::MemoryInfo PjRtComputationClient::GetMemoryInfo(
    const std::string& device) {
  XLA_CHECK_NE(device, spmd_device_str)
      << "MemoryInfo not supported for SPMD virtual device.";
  xla::PjRtDevice* pjrt_device =
      PjRtComputationClient::StringToPjRtDevice(device);
  tsl::AllocatorStats stats = pjrt_device->GetAllocatorStats().value();

  return {
      stats.bytes_in_use,
      *stats.bytes_limit,
      stats.peak_bytes_in_use,
  };
}

void PjRtComputationClient::RegisterCustomCall(const std::string& fn_name,
                                               void* function_ptr,
                                               const std::string& platform) {
  if (platform != "CUDA") {
    XLA_ERROR() << "Custom call targets can only be registered for "
                   "PJRT CUDA runtime.";
    return;
  }

  auto* c_api_client = dynamic_cast<xla::PjRtCApiClient*>(client_.get());
  if (!c_api_client) {
    XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(fn_name, function_ptr, platform);
    return;
  }
  const PJRT_Api* pjrt_api = c_api_client->pjrt_c_api();

  // See openxla reference:
  // https://github.com/openxla/xla/blob/b604c8d87df842002a7a8de79a434026329fbcb2/xla/pjrt/c/pjrt_c_api_gpu_test.cc#L414
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(pjrt_api->extension_start);
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  XLA_CHECK(next) << "Custom call extension not found";
  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = fn_name.size();
  args.api_version = 0;
  args.handler_execute = function_ptr;
  PJRT_Error* error =
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args);
  if (error) {
    XLA_ERROR() << error->status;
  }
}

void PjRtComputationClient::OnReadyCallback(
    ComputationClient::DataPtr data, const std::function<void()>& callback) {
  std::shared_ptr<xla::PjRtBuffer> buffer;
  if (auto pjrt_data = std::dynamic_pointer_cast<PjRtData>(data)) {
    buffer = pjrt_data->buffer;
  } else if (auto sharded_data =
                 std::dynamic_pointer_cast<PjRtShardedData>(data)) {
    XLA_CHECK(sharded_data->shards.size()) << "sharded data has no shards";
    buffer = sharded_data->shards[0]->buffer;
  } else {
    XLA_ERROR() << "received invalid data pointer";
  }
  XLA_CHECK(buffer) << "received placeholder data as argument";
  buffer->GetReadyFuture().OnReady(
      [callback](absl::Status unused) { callback(); });
}

}  // namespace runtime
}  // namespace torch_xla
