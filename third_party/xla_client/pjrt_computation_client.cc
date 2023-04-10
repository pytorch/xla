#include "third_party/xla_client/pjrt_computation_client.h"

#include <algorithm>
#include <unordered_set>

#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "pjrt_computation_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/tf_logging.h"

namespace xla {

namespace {

// Initializes a distributed runtime client if dist_service_addr is specified
std::shared_ptr<DistributedRuntimeClient>
MaybeInitializeDistributedRuntimeClient(int local_rank,
                                        std::string dist_service_addr) {
  std::shared_ptr<DistributedRuntimeClient> client;
  if (!dist_service_addr.empty()) {
    xla::DistributedRuntimeClient::Options options;
    /* TODO(jonbolin): Use global rank for multi-host setup */
    options.node_id = local_rank;
    client =
        xla::GetDistributedRuntimeClient(dist_service_addr, options,
                                         /*use_coordination_service=*/false);
    XLA_CHECK(client->Connect().ok())
        << "Failed to initialize distributed runtime client";
  }
  return std::move(client);
}

}  // namespace

std::string PjRtComputationClient::PjRtDeviceToString(PjRtDevice* const device) const {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  int ordinal = global_ordinals_.at(device->id());
  std::string str = absl::StrFormat("%s:%d", platform, ordinal);
  return str;
}

std::vector<std::string> PjRtComputationClient::PjRtDevicesToString(
    absl::Span<PjRtDevice* const> devices) const {
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
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin(
        "tpu", sys_util::GetEnvString(env::kEnvTpuLibraryPath, "libtpu.so")));
    supports_logical_on_device_shape_ = false;
    client_ = std::move(xla::GetCApiClient("TPU").value());
  } else if (device_type == "TPU_LEGACY") {
    TF_VLOG(1) << "Initializing PjRt StreamExecutor TPU client...";
    int64_t max_inflight_computations = sys_util::GetEnvInt(
        env::kEnvPjRtTpuMaxInflightComputations, /*defval=*/32);
    client_ = xla::GetTpuClient(max_inflight_computations).value();
  } else if (device_type == "GPU") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_rank = sys_util::GetEnvInt(env::kEnvPjRtLocalRank, 0);
    std::string dist_service_addr =
        sys_util::GetEnvString(env::kEnvPjrtDistServiceAddr, "");
    auto distributed_client =
        MaybeInitializeDistributedRuntimeClient(local_rank, dist_service_addr);
    auto allowed_devices =
        std::make_optional<std::set<int>>(std::set{local_rank});
    client_ =
        std::move(xla::GetStreamExecutorGpuClient(
                      /*asynchronous=*/async, GpuAllocatorConfig{},
                      /*distributed_client=*/distributed_client,
                      /*node_id=*/local_rank, allowed_devices = allowed_devices)
                      .value());
  } else {
    XLA_ERROR() << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                   device_type);
  }

  XLA_CHECK(client_.get() != nullptr);

  // PjRtDevice IDs are not guaranteed to be dense, so we need to track
  // a device's global ordinal separately from its device ID. Order the
  // devices by increasing ID to assign global ordinals.
  std::vector<PjRtDevice*> ordered_devices(client_->device_count());
  std::partial_sort_copy(client_->devices().begin(), client_->devices().end(),
    ordered_devices.begin(), ordered_devices.end(),
    [](auto &a, auto &b) { return a->id() < b->id(); });
  for (auto* device : ordered_devices) {
    global_ordinals_[device->id()] = global_ordinals_.size();
    std::string device_str = PjRtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
    device_locks_.emplace(device_str, std::make_unique<std::shared_mutex>());
  }
}

void PjRtComputationClient::PjRtData::Assign(const Data& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtComputationClient::CreateDataPlaceholder(
    std::string device, Shape shape) {
  return std::make_shared<PjRtData>(device, shape);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::GetDataShards(
    ComputationClient::DataPtr data) {
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
    absl::Span<const TensorSource> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::TransferToServer",
      tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  int64_t total_size = 0;
  for (auto& tensor : tensors) {
    PjRtDevice* pjrt_device = StringToPjRtDevice(tensor.device);

    auto literal = std::make_shared<xla::Literal>(tensor.shape);
    tensor.populate_fn(tensor, literal->untyped_data(), literal->size_bytes());
    std::vector<int64_t> byte_strides(literal->shape().dimensions_size());
    XLA_CHECK_OK(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    total_size += literal->size_bytes();

    // Avoid use-after-free on `literal` due to unsequenced move and use.
    xla::Literal* literal_pointer = literal.get();
    std::shared_ptr<xla::PjRtBuffer> buffer = std::move(
        client_
            ->BufferFromHostBuffer(
                literal_pointer->untyped_data(),
                literal_pointer->shape().element_type(),
                literal_pointer->shape().dimensions(), byte_strides,
                xla::PjRtClient::HostBufferSemantics::
                    kImmutableUntilTransferCompletes,
                [literal{std::move(literal)}]() { /* frees literal */ },
                pjrt_device)
            .value());

    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensor.device, tensor.shape, buffer);
    datas.push_back(data);
  }
  OutboundDataMetric()->AddSample(total_size);
  CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr PjRtComputationClient::TransferShardsToServer(
    absl::Span<const TensorSource> tensor_shards, std::string device,
    xla::Shape shape, xla::OpSharding sharding) {
  TF_VLOG(1) << "TransferShardsToServer with " << tensor_shards.size()
             << " shards.";
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
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::CopyToDevice",
      tsl::profiler::TraceMeLevel::kInfo);
  const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(data.get());
  XLA_CHECK(pjrt_data->HasValue()) << "Can't copy invalid device data.";

  PjRtDevice* dst_device = StringToPjRtDevice(dst);
  XLA_CHECK(dst_device->IsAddressable()) << dst << "is not addressable.";

  // Returns error if the buffer is already on `dst_device`.
  StatusOr<std::unique_ptr<PjRtBuffer>> status_or =
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
    TF_VLOG(1) << "ReplicateShardedData (handle=" << handle->GetOpaqueHandle()
               << ", shape=" << handle->shape() << ")";
    xla::XlaBuilder b("ReplicateShardedData");
    xla::Shape shape = sharded_data->shape();
    b.SetSharding(sharded_data->GetSharding());

    // perform a simple identity calculation to reassemble the input as
    // replicated output.
    auto x = xla::Parameter(&b, 0, shape, "p0");
    b.SetSharding(xla::HloSharding::Replicate().ToProto());
    auto y = xla::Div(x, ConstantR0<float>(&b, 2));
    auto z = xla::Add(y, y);

    xla::XlaComputation computation =
        ConsumeValue(b.Build(/*remove_dynamic_dimensions=*/false));
    xla::ProgramShape program_shape =
        ConsumeValue(computation.GetProgramShape());

    std::string device = GetDefaultDevice();
    std::vector<xla::ComputationClient::CompileInstance> instances;
    instances.push_back({std::move(computation), device,
                         GetCompilationDevices(device, {}), &shape,
                         /*should_wrap_parameter=*/false,
                         /*is_sharded=*/true});
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations = Compile(std::move(instances));

    auto shards = sharded_data->shards;
    XLA_CHECK_EQ(shards.size(), GetLocalDevices().size());
    std::vector<std::vector<ComputationClient::DataPtr>> arguments_by_device(
        GetLocalDevices().size(), std::vector<ComputationClient::DataPtr>(1));
    for (auto shard : shards) {
      std::vector<std::string> device_spec =
          absl::StrSplit(shard->device(), ':');
      XLA_CHECK_EQ(device_spec.size(), 2)
          << "Invalid device specification: " << shard->device();
      int device_i = std::stoi(device_spec[1]);
      arguments_by_device[device_i][0] = shard;
    }
    xla::ComputationClient::ExecuteReplicatedOptions execute_options;
    return ExecuteReplicated(*computations.front(), arguments_by_device,
                             GetLocalDevices(), execute_options)[0][0];
  }
  return handle;
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::TransferFromServer",
      tsl::profiler::TraceMeLevel::kInfo);
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  int64_t total_size = 0;
  for (auto handle : handles) {
    // Use XLA replication to reassemble the sharded data. If input handle
    // is not sharded, then it is a no-op.
    auto new_handle = ReplicateShardedData(handle);
    const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(*new_handle);

    // TODO(wcromar): Only use logical_on_device_shape when PJRT C API supports
    // it.
    xla::Shape target_shape = ShapeUtil::DeviceShapeToHostShape(
        supports_logical_on_device_shape_
            ? pjrt_data.buffer->logical_on_device_shape().value()
            : pjrt_data.buffer->on_device_shape());
    auto& literal = literals.emplace_back(target_shape);

    // PJRT will always try to copy the full bounded size into our literal. If
    // the bounded size is larger than the logical output size, we have to
    // allocate a bounded-size literal and copy a slice of the values into our
    // output literal.
    if (pjrt_data.buffer->on_device_shape().is_static()) {
      XLA_CHECK_OK(pjrt_data.buffer->ToLiteralSync(&literal));
    } else {
      std::shared_ptr<xla::Literal> bounded_literal =
          pjrt_data.buffer->ToLiteralSync().value();
      XLA_CHECK_OK(literal.CopySliceFrom(
          *bounded_literal,
          /*src_base=*/std::vector<int64_t>(target_shape.rank(), 0),
          /*dest_base=*/std::vector<int64_t>(target_shape.rank(), 0),
          /*copy_size=*/target_shape.dimensions()));
    }
    total_size += literal.size_bytes();
  }
  InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::Compile",
      tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::ComputationPtr> computations;

  for (auto& instance : instances) {
    xla::CompileOptions compile_options;
    if (instance.is_sharded) {
      // TODO(yeounoh) multi-host, multi-slice configurations
      compile_options.executable_build_options.set_use_spmd_partitioning(true);
      // TODO(yeounoh) this is set to false by default, but explicitly set here
      // to expose the knob for future reference. We can override the compiler's
      // default behavior to further optimize parameter sharding in the future.
      compile_options.executable_build_options
          .set_allow_spmd_sharding_propagation_to_output({false});
      compile_options.executable_build_options.set_num_partitions(
          client_->device_count());
      compile_options.executable_build_options.set_num_replicas(1);
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

      // TODO(244391366) verify this is correct for the collectives ops
      xla::DeviceAssignment device_assignment(1, client_->device_count());
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto &[device_id, global_ordinal] : global_ordinals_) {
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
      for (const auto &[device_id, global_ordinal] : global_ordinals_) {
        device_assignment(global_ordinal, 0) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    }

    PjRtDevice* pjrt_device = StringToPjRtDevice(instance.compilation_device);
    std::unique_ptr<xla::PjRtLoadedExecutable> executable =
        ConsumeValue(client_->Compile(instance.computation, compile_options));

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    HloComputation* hlo_computation = hlo_modules[0]->entry_computation();
    xla::ProgramShape program_shape =
        xla::ProgramShape(hlo_computation->ToProto().program_shape());

    std::shared_ptr<PjRtComputation> pjrt_computation =
        std::make_shared<PjRtComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            program_shape, instance.devices, std::move(executable));

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
  tsl::profiler::TraceMe activity(
      "PjRtComputationClient::ExecuteComputation",
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

  std::optional<PjRtFuture<Status>> returned_future;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
      pjrt_computation.executable
          ->ExecuteSharded(buffers, pjrt_device, execute_options,
                           returned_future)
          .value();

  // Grab the shared lock and block the `WaitDeviceOps` until buffer is ready.
  auto lock = lock_device_shared(device);
  // Signal that `ExecuteSharded` has completed for the ExecuteTime metric.
  // Copies the `timed` shared pointer into the lambda.
  returned_future->OnReady(
      [timed, lock = std::move(lock)](Status unused) mutable {
        timed.reset();
        TF_VLOG(3) << "ExecuteComputation returned_future->OnReady finished";
      });

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

    std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
        device, buffer->on_device_shape(), std::move(buffer));

    datas.push_back(data);
  }
  CreateDataHandlesCounter()->AddValue(datas.size());

  TF_VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

std::vector<std::vector<ComputationClient::DataPtr>>
PjRtComputationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    const std::vector<std::vector<ComputationClient::DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);
  XLA_CHECK(devices.size() == arguments.size())
      << "ExecuteReplicated over " << devices.size() << " devices, but "
      << arguments.size() << " arguments devices.";

  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  for (int32_t i = 0; i < devices.size(); ++i) {
    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[i]);
    XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

    std::vector<PjRtBuffer*> buffers;
    for (auto& argument : arguments[i]) {
      const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

      XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
          << pjrt_device->DebugString() << " vs "
          << pjrt_data->buffer->device()->DebugString();
      buffers.push_back(pjrt_data->buffer.get());
    }
    argument_handles.push_back(buffers);
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = true;
  // TODO(yeounoh) currently only support single-slice execution
  execute_options.multi_slice_config = nullptr;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;  

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      pjrt_computation.executable->Execute(argument_handles, execute_options)
          .value();

  std::vector<std::vector<ComputationClient::DataPtr>> data_handles;
  data_handles.reserve(results.size());
  std::vector<size_t> dims(results.size());
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

      std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
          devices[i], buffer->on_device_shape(), std::move(buffer));
      datas.push_back(data);
    }
    data_handles.push_back(datas);
  }

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

const absl::flat_hash_map<std::string, xla::ComputationClient::DeviceAttribute>&
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

}  // namespace xla
