#include "torch_xla/csrc/runtime/pjrt_compilation_client.h"

#include <algorithm>
#include <future>
#include <unordered_set>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_hash.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/operation_manager.h"
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
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/protobuf_util.h"
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

torch::lazy::hash_t hash_comp_env() {
  // TODO(piz): since the client is nullptr, we can't retrive all information
  // like PjRtComputationClient. Think about a way to construct the hashing.
  torch::lazy::hash_t hash = hash::HashXlaEnvVars();
  return hash;
}

}  // namespace

std::string PjRtCompilationClient::PjRtDeviceToString(
    xla::PjRtDevice* const device) const {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  int ordinal = global_ordinals_.at(device->id());
  std::string str = absl::StrFormat("%s:%d", platform, ordinal);
  return str;
}

std::vector<std::string> PjRtCompilationClient::PjRtDevicesToString(
    absl::Span<xla::PjRtDevice* const> devices) const {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(PjRtDeviceToString(device));
  }

  return strs;
}

PjRtCompilationClient::PjRtCompilationClient(
    std::string& virtual_topology_str) {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");

  auto tpu_library_path = sys_util::GetEnvString(
      env::kEnvTpuLibraryPath,
      sys_util::GetEnvString(env::kEnvInferredTpuLibraryPath, "libtpu.so"));
  XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
  xla::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
  XLA_CHECK_OK(tpu_status);

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {};

  // TODO(piz): we need to specify correct replicas and partitions
  absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>> topo =
      xla::GetCApiTopology("tpu", virtual_topology_str, create_options);
  XLA_CHECK_OK(topo.status());
  this->virtual_topology = std::move(topo.value());

  // parsing the fake topology
  // TODO(piz): this is a temporary solution to convert the topology into
  // devices. Fix this for SPMD case.
  std::string device_topology;
  size_t pos = virtual_topology_str.find(':');
  if (pos != std::string::npos) {
    device_topology = virtual_topology_str.substr(pos + 1);
  }

  size_t pre_pos = 0;
  int device_count = 1;
  do {
    pos = device_topology.find('x', pre_pos);
    int topo_dim = std::stoi(device_topology.substr(pre_pos, pos - pre_pos));
    device_count *= topo_dim;
    pre_pos = pos + 1;
  } while (pos != std::string::npos);

  for (int i = 0; i < device_count; i++) {
    this->client_addressable_devices.push_back(device_type + ":" +
                                               std::to_string(i));
    this->client_devices.push_back(device_type + std::to_string(i));
  }
  client_addressable_device_count = this->client_addressable_devices.size();
  client_device_count = this->client_devices.size();

  // PjRtDevice IDs are not guaranteed to be dense, so we need to track
  // a device's global ordinal separately from its device ID. Order the
  // devices by increasing ID to assign global ordinals.
  for (size_t i = 0; i < this->client_device_count; i++) {
    global_ordinals_[i] = global_ordinals_.size();
  }

  comp_env_hash_ = hash_comp_env();

  auto tracked_devices = GetLocalDevices();
  tracked_devices.emplace_back(spmd_device_str);
  operation_manager_ = std::move(OperationManager(std::move(tracked_devices)));
}

PjRtCompilationClient::~PjRtCompilationClient() {
  // In the GPU case, the PjRtClient depends on the DistributedRuntimeClient
  // tracked in XlaCoordinator, so the PjRtClient must be destroyed first.
  client_ = nullptr;
  coordinator_ = nullptr;
}

bool PjRtCompilationClient::CoordinatorInitialized() const {
  return coordinator_ != nullptr;
}

void PjRtCompilationClient::InitializeCoordinator(int global_rank,
                                                  int world_size,
                                                  std::string master_addr,
                                                  std::string port) {
  XLA_CHECK(coordinator_ == nullptr)
      << "Can only initialize the XlaCoordinator once.";
  coordinator_ = std::make_unique<XlaCoordinator>(global_rank, world_size,
                                                  master_addr, port);
}

XlaCoordinator& PjRtCompilationClient::GetCoordinator() {
  XLA_CHECK(coordinator_ != nullptr)
      << "XlaCoordinator has not been initialized";
  return *coordinator_;
}

void PjRtCompilationClient::PjRtData::Assign(
    const torch::lazy::BackendData& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtCompilationClient::CreateDataPlaceholder(
    std::string device, xla::Shape shape,
    std::optional<xla::OpSharding> sharding) {
  if (sharding.has_value()) {
    return std::make_shared<PjRtShardedData>(
        std::move(device), std::move(shape), std::move(*sharding));
  }

  return std::make_shared<PjRtData>(std::move(device), std::move(shape));
}

ComputationClient::DataPtr PjRtCompilationClient::CreateData(
    std::string device, xla::Shape shape, std::shared_ptr<Buffer> buffer) {
  return std::make_shared<PjRtData>(std::move(device), std::move(shape),
                                    buffer);
}

std::vector<ComputationClient::DataPtr> PjRtCompilationClient::GetDataShards(
    ComputationClient::DataPtr data) {
  tsl::profiler::TraceMe activity("PjRtCompilationClient::GetDataShards",
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

ComputationClient::DataPtr PjRtCompilationClient::GetDataShard(
    ComputationClient::DataPtr data, size_t index) {
  tsl::profiler::TraceMe activity("PjRtCompilationClient::GetDataShard",
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

ComputationClient::DataPtr PjRtCompilationClient::WrapDataShards(
    absl::Span<const DataPtr> shards, std::string device, xla::Shape shape,
    xla::OpSharding sharding) {
  XLA_CHECK_EQ(shards.size(), client_addressable_devices.size());
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

std::optional<xla::OpSharding> PjRtCompilationClient::GetDataSharding(
    DataPtr handle) {
  if (auto sharded_data = dynamic_cast<PjRtShardedData*>(handle.get())) {
    return sharded_data->GetSharding();
  }
  return std::optional<xla::OpSharding>();
}

std::vector<ComputationClient::DataPtr> PjRtCompilationClient::TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  int64_t total_size = 0;
  for (auto& tensor : tensors) {
    total_size += xla::ShapeUtil::ByteSizeOf(tensor->shape());
    std::vector<xla::Shape> tuple_shape;
    absl::Span<const bool> dynamic_dimensions;
    xla::Shape shape(tensor->primitive_type(), tensor->dimensions(),
                     dynamic_dimensions, tuple_shape);
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(shape);
    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensor->device(), tensor->shape(), buffer);
    datas.push_back(data);
  }
  OutboundDataMetric()->AddSample(total_size);
  CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr PjRtCompilationClient::TransferShardsToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
    std::string device, xla::Shape shape, xla::OpSharding sharding) {
  tsl::profiler::TraceMe activity(
      "PjRtCompilationClient::TransferShardsToDevice",
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

ComputationClient::DataPtr PjRtCompilationClient::CopyToDevice(
    ComputationClient::DataPtr data, std::string dst) {
  tsl::profiler::TraceMe activity("PjRtCompilationClient::CopyToDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(data.get());
  XLA_CHECK(pjrt_data->HasValue()) << "Can't copy invalid device data.";

  xla::PjRtDevice* dst_device = StringToPjRtDevice(dst);
  XLA_CHECK(dst_device->IsAddressable()) << dst << "is not addressable.";
  return std::make_shared<PjRtData>(dst, pjrt_data->shape(), pjrt_data->buffer);
}

std::shared_ptr<PjRtCompilationClient::PjRtData>
PjRtCompilationClient::ReplicateShardedData(
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

std::vector<ComputationClient::DataPtr> PjRtCompilationClient::ReshardData(
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

std::uintptr_t PjRtCompilationClient::UnsafeBufferPointer(
    const DataPtr handle) {
  TF_VLOG(3) << "UnsafeBufferPointer is not umplemented for compilation only";
  return 0;
}

std::shared_ptr<xla::PjRtBuffer> PjRtCompilationClient::GetPjRtBuffer(
    const DataPtr handle) {
  TF_LOG(ERROR) << "AOT compilation is unable to get buffer data from device";
  return std::shared_ptr<xla::PjRtBuffer>(nullptr);
}

std::vector<xla::Literal> PjRtCompilationClient::TransferFromDevice(
    absl::Span<const DataPtr> handles) {
  TF_LOG(ERROR) << "AOT compilation is unable to run compuatation and transfer "
                   "data from device";
  std::vector<xla::Literal> literals;
  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtCompilationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  tsl::profiler::TraceMe activity("PjRtCompilationClient::Compile",
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

      int num_partitions = client_device_count;

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
      xla::DeviceAssignment device_assignment(1, client_device_count);
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
          client_device_count);
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

      xla::DeviceAssignment device_assignment(client_device_count, 1);
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto& [device_id, global_ordinal] : global_ordinals_) {
        device_assignment(global_ordinal, 0) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    }

    std::shared_ptr<xla::PjRtTopologyDescription> topo =
        std::move(this->virtual_topology);
    std::unique_ptr<xla::PjRtExecutable> executable = ConsumeValue(
        PjRtCompile(compile_options, instance.computation, *topo.get()));
    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    xla::HloComputation* hlo_computation = hlo_modules[0]->entry_computation();
    std::shared_ptr<PjRtUnloadedComputation> pjrt_computation =
        std::make_shared<PjRtUnloadedComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            instance.devices, std::move(executable));
    computations.push_back(pjrt_computation);
    CreateCompileHandlesCounter()->AddValue(1);
  }

  return computations;
}

std::string PjRtCompilationClient::SerializeComputation(
    const ComputationPtr computation) {
  // AOT uses PjRtUnloadedComputation, which doesn't need a client
  const PjRtUnloadedComputation& pjrt_computation =
      dynamic_cast<const PjRtUnloadedComputation&>(*computation);
  return ConsumeValue(pjrt_computation.executable->SerializeExecutable());
}

ComputationClient::ComputationPtr PjRtCompilationClient::DeserializeComputation(
    const std::string& serialized) {
  TF_LOG(ERROR) << __FUNCTION__ << " is not defined for AOT compilation";
  return nullptr;
}

torch::lazy::hash_t PjRtCompilationClient::HashCompilationEnv() {
  // TODO(jonbolin): Incorporate CompileOptions into the hash. These are
  // deterministically generated at the moment, so they don't need to be
  // included. It will require a small refactor, so punting on this for now.
  return comp_env_hash_;
}

std::vector<ComputationClient::DataPtr>
PjRtCompilationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  TF_LOG(ERROR) << __FUNCTION__ << " is not supported for AOT compilation";
  return std::vector<ComputationClient::DataPtr>();
}

std::vector<ComputationClient::DataPtr>
PjRtCompilationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  TF_LOG(ERROR) << __FUNCTION__ << " is not supported for AOT compilation";
  std::vector<ComputationClient::DataPtr> data_handles;
  return data_handles;
}

size_t PjRtCompilationClient::GetNumDevices() const {
  return this->client_addressable_device_count;
}

std::string PjRtCompilationClient::GetDefaultDevice() const {
  return this->client_addressable_devices[0];
}

std::vector<std::string> PjRtCompilationClient::GetLocalDevices() const {
  return this->client_addressable_devices;
}

std::vector<std::string> PjRtCompilationClient::GetAllDevices() const {
  return this->client_devices;
}

int PjRtCompilationClient::GetNumProcesses() const {
  TF_LOG(ERROR) << __FUNCTION__ << " is not defined for AOT compilation";
  return 1;
};

const absl::flat_hash_map<
    std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>&
PjRtCompilationClient::GetDeviceAttributes(const std::string& device) {
  return PjRtCompilationClient::StringToPjRtDevice(device)->Attributes();
}

void PjRtCompilationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
PjRtCompilationClient::GetReplicationDevices() {
  return replication_devices_;
}

xla::PjRtDevice* PjRtCompilationClient::StringToPjRtDevice(
    const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
      << "Unknown device " << device;
  xla::PjRtDevice* pjrt_device = string_to_device_[device];
  return pjrt_device;
}

void PjRtCompilationClient::WaitDeviceOps(
    absl::Span<const std::string> devices) {
  TF_VLOG(3) << "Waiting for " << absl::StrJoin(devices, ", ");
  operation_manager_.WaitForDevices(devices.empty() ? GetLocalDevices()
                                                    : devices);
}

std::map<std::string, Metric> PjRtCompilationClient::GetMetrics() const {
  // TODO(jonbolin): Add any PJRt-client-specific metrics here
  return {};
}

ComputationClient::MemoryInfo PjRtCompilationClient::GetMemoryInfo(
    const std::string& device) {
  XLA_CHECK_NE(device, spmd_device_str)
      << "MemoryInfo not supported for SPMD virtual device.";
  xla::PjRtDevice* pjrt_device =
      PjRtCompilationClient::StringToPjRtDevice(device);
  tsl::AllocatorStats stats = pjrt_device->GetAllocatorStats().value();

  return {
      stats.bytes_in_use,
      *stats.bytes_limit,
  };
}

}  // namespace runtime
}  // namespace torch_xla
