#include "torch_xla/csrc/runtime/pjrt_computation_client.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "pjrt_computation_client.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/multi_wait.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/tensor_source.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/thread_pool.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/shape.h"

using xla::internal::XlaBuilderFriend;

namespace torch_xla {
namespace runtime {

namespace {

static std::string spmd_device_str = "SPMD:0";

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
  *shape.mutable_layout() = buffer->layout();

  return xla::ShapeUtil::DeviceShapeToHostShape(shape);
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
  if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncCpuClient, true);
    int cpu_device_count = sys_util::GetEnvInt(env::kEnvNumCpu, 1);
    client_ = std::move(xla::GetTfrtCpuClient(async, cpu_device_count).value());
  } else if (device_type == "TPU" || device_type == "TPU_C_API") {
    TF_VLOG(1) << "Initializing TFRT TPU client...";
    XLA_CHECK_OK(
        pjrt::LoadPjrtPlugin(
            "tpu", sys_util::GetEnvString(env::kEnvTpuLibraryPath, "libtpu.so"))
            .status());
    tsl::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
    XLA_CHECK(tpu_status.ok());
    client_ = std::move(xla::GetCApiClient("TPU").value());
  } else if (device_type == "TPU_LEGACY") {
    XLA_ERROR() << "TPU_LEGACY client is no longer available.";
  } else if (device_type == "GPU" || device_type == "CUDA" ||
             device_type == "ROCM") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_process_rank = sys_util::GetEnvInt(env::kEnvPjRtLocalRank, 0);
    int global_process_rank = sys_util::GetEnvInt("RANK", local_process_rank);
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);
    std::string master_addr =
        runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
    std::string port = runtime::sys_util::GetEnvString(
        "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);

    // Use the XlaCoordinator as the distributed key-value store.
    coordinator_ = std::make_unique<XlaCoordinator>(
        global_process_rank, global_world_size, master_addr, port);
    std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
        coordinator_->GetClient();
    auto allowed_devices =
        std::make_optional<std::set<int>>(std::set{local_process_rank});
    xla::PjRtClient::KeyValueGetCallback kv_get = nullptr;
    xla::PjRtClient::KeyValuePutCallback kv_put = nullptr;
    if (distributed_client != nullptr) {
      std::string key_prefix = "gpu:";
      kv_get = [distributed_client, key_prefix](const std::string& k,
                                                absl::Duration timeout) {
        return distributed_client->BlockingKeyValueGet(
            absl::StrCat(key_prefix, k), timeout);
      };
      kv_put = [distributed_client, key_prefix](const std::string& k,
                                                const std::string& v) {
        return distributed_client->KeyValueSet(absl::StrCat(key_prefix, k), v);
      };
    }
    TF_VLOG(3) << "Getting StreamExecutorGpuClient for node_id="
               << global_process_rank << ", num_nodes=" << global_world_size;
    client_ = std::move(xla::GetStreamExecutorGpuClient(
                            /*asynchronous=*/async,
                            /*allocator_config=*/xla::GpuAllocatorConfig{},
                            /*node_id=*/global_process_rank,
                            /*num_nodes=*/global_world_size,
                            /*allowed_devices=*/allowed_devices,
                            /*platform_name=*/"gpu",
                            /*should_stage_host_to_device_transfers=*/true,
                            /*kv_get=*/kv_get,
                            /*kv_put=*/kv_put)
                            .value());
  } else if (device_type == "XPU") {
    TF_VLOG(1) << "Initializing PjRt XPU client...";
    XLA_CHECK_OK(
        pjrt::LoadPjrtPlugin(
            "xpu", sys_util::GetEnvString(env::kEnvXpuLibraryPath, "libxpu.so"))
            .status());
    client_ = std::move(xla::GetCApiClient("XPU").value());
  } else if (device_type == "NEURON") {
    TF_VLOG(1) << "Initializing PjRt NEURON client...";
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("NEURON", sys_util::GetEnvString(
                                                    env::kEnvNeuronLibraryPath,
                                                    "libneuronpjrt.so"))
                     .status());
    client_ = std::move(xla::GetCApiClient("NEURON").value());
  } else {
    XLA_ERROR() << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                   device_type);
  }

  XLA_CHECK(client_.get() != nullptr);

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
    device_locks_.emplace(device_str, std::make_unique<std::shared_mutex>());
  }
  // manually create the device_locks for SPMD device
  device_locks_.emplace(spmd_device_str, std::make_unique<std::shared_mutex>());
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
    std::string device, xla::Shape shape) {
  return std::make_shared<PjRtData>(device, shape);
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
    const std::vector<DataPtr>& shards, std::string device, xla::Shape shape,
    xla::OpSharding sharding) {
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

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToServer(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());
  tsl::profiler::TraceMe activity("PjRtComputationClient::TransferToServer",
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
                          tensor->data(), tensor->shape().element_type(),
                          tensor->shape().dimensions(), tensor->byte_strides(),
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

ComputationClient::DataPtr PjRtComputationClient::TransferShardsToServer(
    absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
    std::string device, xla::Shape shape, xla::OpSharding sharding) {
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::TransferShardsToServer",
      tsl::profiler::TraceMeLevel::kInfo);
  // TODO(jonbolin): Consider using CopyToDevice when sharding is REPLICATED.
  // We are opting out of CopyToDevice for now due to the synchronization
  // issues observed in ShardingUtil::InputHandler, but because CopyToDevice
  // directly copies buffers between devices using ICI, it can be much faster
  // than transferring from the host to each device.
  auto data_shards = TransferToServer(tensor_shards);
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
  xla::StatusOr<std::unique_ptr<xla::PjRtBuffer>> status_or =
      pjrt_data->buffer->CopyToDevice(dst_device);
  XLA_CHECK(status_or.ok())
      << pjrt_data->device() << " buffer already exists on " << dst;

  return std::make_shared<PjRtData>(dst, pjrt_data->shape(),
                                    std::move(status_or.value()));
}

ComputationClient::DataPtr PjRtComputationClient::ReplicateShardedData(
    const ComputationClient::DataPtr& handle) {
  if (PjRtShardedData* sharded_data =
          dynamic_cast<PjRtShardedData*>(handle.get())) {
    XLA_COUNTER("ReplicateShardedData", 1);
    TF_VLOG(1) << "ReplicateShardedData (handle=" << handle->GetHandle()
               << ", shape=" << handle->shape() << ")";
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

    auto shards = sharded_data->shards;
    XLA_CHECK_EQ(shards.size(), GetLocalDevices().size());
    auto device_index = build_index_map(GetLocalDevices());

    std::vector<std::vector<ComputationClient::DataPtr>> arguments_by_device(
        GetLocalDevices().size(), std::vector<ComputationClient::DataPtr>(1));
    for (auto shard : shards) {
      std::vector<std::string> device_spec =
          absl::StrSplit(shard->device(), ':');
      XLA_CHECK_EQ(device_spec.size(), 2)
          << "Invalid device specification: " << shard->device();
      int device_i = device_index[std::stoi(device_spec[1])];
      TF_VLOG(3) << shard->device() << " is mapped to local device index "
                 << device_i;
      arguments_by_device[device_i][0] = shard;
    }
    torch_xla::runtime::ComputationClient::ExecuteReplicatedOptions
        execute_options;
    auto sharded_results =
        ExecuteReplicated(*computations.front(), arguments_by_device,
                          GetLocalDevices(), execute_options);
    XLA_CHECK(sharded_results.size() > 0)
        << "empty ExecuteReplicated results returned.";
    XLA_CHECK(sharded_results[0].size() == 1)
        << "Wrong number of outputs, expected: 1, actual: "
        << sharded_results[0].size();
    return sharded_results[0][0];
  }
  return handle;
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());
  tsl::profiler::TraceMe activity("PjRtComputationClient::TransferFromServer",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  int64_t total_size = 0;
  for (auto handle : handles) {
    // Use XLA replication to reassemble the sharded data. If input handle
    // is not sharded, then it is a no-op.
    auto new_handle = ReplicateShardedData(handle);
    const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(*new_handle);

    auto& literal =
        literals.emplace_back(host_output_shape(pjrt_data.buffer.get()));
    XLA_CHECK_OK(pjrt_data.buffer->ToLiteralSync(&literal));

    total_size += literal.size_bytes();
  }
  InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
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
      compile_options.executable_build_options.set_num_partitions(
          client_->device_count());
      compile_options.executable_build_options.set_num_replicas(1);
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

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
      torch_xla::runtime::ConvertHloToStableHlo(
          instance.computation.mutable_proto(), &mlir_module);
      executable = ConsumeValue(client_->Compile(mlir_module, compile_options));
      StableHloCompileCounter()->AddValue(1);
    } else {
      executable =
          ConsumeValue(client_->Compile(instance.computation, compile_options));
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

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteComputation` and the async work in `ExecuteSharded` are
  // complete; a copy is held from the lambda that releases it when done.
  auto timed = std::make_shared<metrics::TimedSection>(ExecuteMetric());
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
        << pjrt_device->DebugString() << " vs "
        << pjrt_data->buffer->device()->DebugString();
    buffers.push_back(pjrt_data->buffer.get());
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = false;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;

  std::optional<xla::PjRtFuture<xla::Status>> returned_future;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
      pjrt_computation.executable
          ->ExecuteSharded(buffers, pjrt_device, execute_options,
                           returned_future)
          .value();

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

    std::shared_ptr<PjRtData> data =
        std::make_shared<PjRtData>(device, std::move(buffer));

    datas.push_back(data);
  }
  CreateDataHandlesCounter()->AddValue(datas.size());

  auto mwait = std::make_shared<util::MultiWait>(1);
  auto lockfn = [&, this, device, returned_future = std::move(*returned_future),
                 timed]() mutable {
    TF_VLOG(5) << "ExecuteComputation acquiring PJRT device lock for "
               << device;
    // Grab the shared lock and block the `WaitDeviceOps` until buffer is
    // ready.
    // TODO(JackCaoG): This lock should acquired outside of the lockfn and
    // passed in. It is possible that lockfn started after ExecuteComputation
    // released the xla_graph_executor lock, which will create a short windows
    // where device is unlcoked while execution is still running.
    auto lock = lock_device_shared(device);
    TF_VLOG(5) << "ExecuteComputation acquiring PJRT device lock for " << device
               << " Done";
    // Signal that `ExecuteSharded` has completed for the ExecuteTime
    // metric. Copies the `timed` shared pointer into the lambda.
    XLA_CHECK(returned_future.IsValid())
        << "returned_future in ExecuteComputation is empty";
    returned_future.OnReady(
        [timed, lock = std::move(lock)](xla::Status unused) mutable {
          timed.reset();
          TF_VLOG(3) << "ExecuteComputation returned_future->OnReady finished";
        });
  };

  env::ScheduleIoClosure(util::MultiWait::Completer(mwait, std::move(lockfn)));

  TF_VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

std::vector<std::vector<ComputationClient::DataPtr>>
PjRtComputationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    const std::vector<std::vector<ComputationClient::DataPtr>>& arguments,
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
  XLA_CHECK(devices.size() == arguments.size())
      << "ExecuteReplicated over " << devices.size() << " devices, but "
      << arguments.size() << " arguments devices.";
  auto mwait_argument = std::make_shared<util::MultiWait>(devices.size());
  std::vector<std::vector<xla::PjRtBuffer*>> argument_handles(devices.size());
  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_argument_handle",
        tsl::profiler::TraceMeLevel::kInfo);
    for (int32_t i = 0; i < devices.size(); ++i) {
      auto buffer_converter = [&, i]() {
        xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[i]);
        XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

        std::vector<xla::PjRtBuffer*> buffers;
        for (auto& argument : arguments[i]) {
          const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

          XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
              << pjrt_device->DebugString() << " vs "
              << pjrt_data->buffer->device()->DebugString();
          buffers.push_back(pjrt_data->buffer.get());
        }
        argument_handles[i] = std::move(buffers);
      };
      env::ScheduleIoClosure(util::MultiWait::Completer(
          mwait_argument, std::move(buffer_converter)));
    }
    mwait_argument->Wait();
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = true;
  // TODO(yeounoh) currently only support single-slice execution
  execute_options.multi_slice_config = nullptr;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;

  std::optional<std::vector<xla::PjRtFuture<xla::Status>>> returned_futures(
      devices.size());
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results;
  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_execute",
        tsl::profiler::TraceMeLevel::kInfo);
    results = pjrt_computation.executable
                  ->Execute(std::move(argument_handles), execute_options,
                            returned_futures)
                  .value();
  }

  std::vector<std::vector<ComputationClient::DataPtr>> data_handles;
  data_handles.reserve(results.size());
  std::vector<size_t> dims(results.size());

  {
    tsl::profiler::TraceMe activity(
        "PjRtComputationClient::ExecuteReplicated_result_handle",
        tsl::profiler::TraceMeLevel::kInfo);
    for (int32_t i = 0; i < results.size(); ++i) {
      xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[i]);
      XLA_CHECK(pjrt_device->IsAddressable())
          << pjrt_device->DebugString() << " is not addressable.";

      std::vector<ComputationClient::DataPtr> datas;
      datas.reserve(results[i].size());
      dims[i] = results[i].size();
      for (int32_t j = 0; j < results[i].size(); ++j) {
        std::unique_ptr<xla::PjRtBuffer> buffer = std::move(results[i][j]);
        XLA_CHECK(pjrt_device == buffer->device())
            << "Exepcted device: " << pjrt_device->DebugString()
            << " vs. actual device: " << buffer->device()->DebugString();

        std::shared_ptr<PjRtData> data =
            std::make_shared<PjRtData>(devices[i], std::move(buffer));
        datas.push_back(data);
      }
      data_handles.push_back(datas);
    }
  }

  auto mwait = std::make_shared<util::MultiWait>(1);
  auto lockfn = [&, this, returned_futures = std::move(*returned_futures),
                 timed]() mutable {
    // Grab the shared lock and block the `WaitDeviceOps` until buffer is
    // ready. Since this is the SPMD code path. There is no points to grab
    // devices lock for every individual device.
    TF_VLOG(5) << "ExecuteReplicated acquiring PJRT device lock for "
               << spmd_device_str;
    auto lock = lock_device_shared(spmd_device_str);
    TF_VLOG(5) << "ExecuteReplicated acquiring PJRT device lock for "
               << spmd_device_str << " Done";
    // Signal that `ExecuteReplicated` has completed for one of the devices
    // the ExecuteReplicatedTime metric. Here, we assume that all devices
    // will finish execution roughly at the same time, hence only use one of
    // the returned_futures. Copies the `timed` shared pointer into the
    // lambda.
    XLA_CHECK(returned_futures[0].IsValid())
        << "returned_future in ExecuteReplicated is empty";
    returned_futures[0].OnReady(
        [timed, lock = std::move(lock)](xla::Status unused) mutable {
          timed.reset();
          TF_VLOG(3) << "ExecuteReplicated returned_future->OnReady finished";
        });
  };
  env::ScheduleIoClosure(util::MultiWait::Completer(mwait, std::move(lockfn)));

  TF_VLOG(1) << "Returning " << data_handles.size() << " sets of results "
             << "with dimensions [" << absl::StrJoin(dims, ",") << "].";
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
    std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>&
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

std::shared_lock<std::shared_mutex> PjRtComputationClient::lock_device_shared(
    const std::string& device) {
  std::shared_lock lock(*device_locks_[device]);
  return lock;
}

std::unique_lock<std::shared_mutex> PjRtComputationClient::lock_device(
    const std::string& device) {
  std::unique_lock lock(*device_locks_[device]);
  return lock;
}

void PjRtComputationClient::WaitDeviceOps(
    const std::vector<std::string>& devices) {
  std::unordered_set<std::string> wait_devices;
  if (!devices.empty()) {
    for (auto& device_str : devices) {
      wait_devices.insert(device_str);
    }
  } else {
    for (auto& device_str : GetLocalDevices()) {
      wait_devices.insert(device_str);
    }
  }
  for (const std::string& device_str : wait_devices) {
    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish";
    lock_device(device_str);
    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish.. Done";
  }
}

std::map<std::string, Metric> PjRtComputationClient::GetMetrics() const {
  // TODO(jonbolin): Add any PJRt-client-specific metrics here
  return {};
}

}  // namespace runtime
}  // namespace torch_xla
