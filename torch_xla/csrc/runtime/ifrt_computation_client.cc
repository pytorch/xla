#include "torch_xla/csrc/runtime/ifrt_computation_client.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_hash.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_registry.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
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

torch::lazy::hash_t hash_comp_env(
    xla::ifrt::Client* client,
    std::vector<xla::ifrt::Device*>& ordered_devices) {
  torch::lazy::hash_t hash = hash::HashXlaEnvVars();
  std::string platform_name(client->platform_name());
  std::string platform_version(client->platform_version());
  hash = torch::lazy::HashCombine(
      hash, torch::lazy::StringHash(platform_name.c_str()));
  // platform_version incorporates libtpu version and hardware type.
  hash = torch::lazy::HashCombine(
      hash, torch::lazy::StringHash(platform_version.c_str()));
  // Include global devices in the hash, ensuring order is consistent.
  xla::ifrt::BasicDeviceList::Devices ifrt_devices;
  for (auto& device : ordered_devices) {
    std::string device_str(device->ToString());
    hash = torch::lazy::HashCombine(
        hash, torch::lazy::StringHash(device_str.c_str()));
    ifrt_devices.push_back(device);
  }

  tsl::RCReference<xla::ifrt::DeviceList> device_list =
      xla::ifrt::BasicDeviceList::Create(std::move(ifrt_devices));

  auto topology_desc = client->GetTopologyForDevices(device_list);
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
  return hash;
}

}  // namespace

std::string IfrtComputationClient::IfrtDeviceToString(
    xla::ifrt::Device* const device) const {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  int ordinal = global_ordinals_.at(device->Id().value());
  std::string str = absl::StrFormat("%s:%d", platform, ordinal);
  return str;
}

std::vector<std::string> IfrtComputationClient::IfrtDevicesToString(
    absl::Span<xla::ifrt::Device* const> devices) const {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(IfrtDeviceToString(device));
  }

  return strs;
}

IfrtComputationClient::IfrtComputationClient() {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");
  std::unique_ptr<xla::PjRtClient> pjrt_client;
  std::tie(pjrt_client, coordinator_) = InitializePjRt(device_type);

  client_ = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));

  // PjRtDevice IDs are not guaranteed to be dense, so we need to track
  // a device's global ordinal separately from its device ID. Order the
  // devices by increasing ID to assign global ordinals.
  std::vector<xla::ifrt::Device*> ordered_devices(client_->device_count());
  std::partial_sort_copy(
      client_->devices().begin(), client_->devices().end(),
      ordered_devices.begin(), ordered_devices.end(),
      [](auto& a, auto& b) { return a->Id().value() < b->Id().value(); });
  for (auto* device : ordered_devices) {
    global_ordinals_[device->Id().value()] = global_ordinals_.size();
    std::string device_str = IfrtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
  }
  comp_env_hash_ = hash_comp_env(client_.get(), ordered_devices);

  auto tracked_devices = GetLocalDevices();
  tracked_devices.emplace_back(spmd_device_str);
  operation_manager_ = std::move(OperationManager(std::move(tracked_devices)));
}

IfrtComputationClient::~IfrtComputationClient() {
  // In the GPU case, the PjRtClient depends on the DistributedRuntimeClient
  // tracked in XlaCoordinator, so the PjRtClient must be destroyed first.
  client_ = nullptr;
  coordinator_ = nullptr;
}

bool IfrtComputationClient::CoordinatorInitialized() const {
  return coordinator_ != nullptr;
}

void IfrtComputationClient::InitializeCoordinator(int global_rank,
                                                  int world_size,
                                                  std::string master_addr,
                                                  std::string port) {
  XLA_CHECK(coordinator_ == nullptr)
      << "Can only initialize the XlaCoordinator once.";
  coordinator_ = std::make_unique<XlaCoordinator>(global_rank, world_size,
                                                  master_addr, port);
}

XlaCoordinator& IfrtComputationClient::GetCoordinator() {
  XLA_CHECK(coordinator_ != nullptr)
      << "XlaCoordinator has not been initialized";
  return *coordinator_;
}

void IfrtComputationClient::IfrtData::Assign(
    const torch::lazy::BackendData& data) {
  const IfrtData& ifrt_data = dynamic_cast<const IfrtData&>(data);
  if (&ifrt_data != this) {
    buffer = ifrt_data.buffer;
  }
}

xla::OpSharding IfrtComputationClient::IfrtData::GetSharding() const {
  XLA_CHECK(HasSharding()) << "Check HasSharding first";
  return *sharding_;
}

ComputationClient::DataPtr IfrtComputationClient::CreateDataPlaceholder(
    std::string device, xla::Shape shape,
    std::optional<xla::OpSharding> sharding) {
  return std::make_shared<IfrtData>(std::move(device), std::move(shape),
                                    tsl::RCReference<xla::ifrt::Array>(),
                                    std::move(sharding));
}

std::vector<ComputationClient::DataPtr> IfrtComputationClient::GetDataShards(
    ComputationClient::DataPtr data) {
  tsl::profiler::TraceMe activity("IfrtComputationClient::GetDataShards",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> shards;
  if (data->HasSharding()) {
    auto ifrt_data = std::dynamic_pointer_cast<IfrtData>(data);
    std::vector<tsl::RCReference<xla::ifrt::Array>> arrays =
        ifrt_data->buffer
            ->DisassembleIntoSingleDeviceArrays(
                xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .value();

    for (auto array : arrays) {
      shards.push_back(std::make_shared<IfrtData>(
          IfrtDeviceToString(array->sharding().devices()->devices().front()),
          array));
    }
  } else {
    shards.push_back(data);
  }
  return shards;
}

ComputationClient::DataPtr IfrtComputationClient::GetDataShard(
    ComputationClient::DataPtr data, size_t index) {
  tsl::profiler::TraceMe activity("IfrtComputationClient::GetDataShard",
                                  tsl::profiler::TraceMeLevel::kInfo);
  return GetDataShards(data)[index];
}

ComputationClient::DataPtr IfrtComputationClient::WrapDataShards(
    absl::Span<const DataPtr> shards, std::string device, xla::Shape shape,
    xla::OpSharding sharding) {
  XLA_CHECK_EQ(shards.size(), client_->addressable_device_count());
  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  std::vector<xla::ifrt::Shape> shard_shapes;
  for (auto& shard : shards) {
    auto ifrt_shard = std::dynamic_pointer_cast<IfrtData>(shard);
    arrays.push_back(ifrt_shard->buffer);
    shard_shapes.push_back(ifrt_shard->buffer->shape());
  }
  xla::ifrt::Shape ifrt_shape(shape.dimensions());
  tsl::RCReference<xla::ifrt::DeviceList> devices_list =
      xla::ifrt::BasicDeviceList::Create(
          {client_->addressable_devices().begin(),
           client_->addressable_devices().end()});

  XLA_CHECK_EQ(shard_shapes.size(), devices_list->size());
  std::unique_ptr<xla::ifrt::Sharding> ifrt_sharding =
      xla::ifrt::ConcreteSharding::Create(devices_list, xla::ifrt::MemoryKind(),
                                          ifrt_shape, shard_shapes);
  // TODO: Attach HloSharding instead when it is supported
  // std::unique_ptr<xla::ifrt::Sharding> ifrt_sharding =
  // xla::ifrt::HloSharding::Create(
  //   devices_list,
  //   xla::ifrt::MemoryKind(),
  //   xla::HloSharding::FromProto(sharding).value()
  // );
  tsl::RCReference<xla::ifrt::Array> sharded_array =
      client_
          ->AssembleArrayFromSingleDeviceArrays(
              ifrt_shape, std::move(ifrt_sharding), absl::MakeSpan(arrays),
              xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
          .value();
  return std::make_shared<IfrtData>(device, shape, sharded_array, sharding);
}

std::optional<xla::OpSharding> IfrtComputationClient::GetDataSharding(
    DataPtr handle) {
  auto ifrt_data = std::dynamic_pointer_cast<IfrtData>(handle);
  return ifrt_data->sharding_;
}

std::vector<ComputationClient::DataPtr> IfrtComputationClient::TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  auto timed =
      std::make_shared<metrics::TimedSection>(TransferToDeviceMetric());
  tsl::profiler::TraceMe activity("IfrtComputationClient::TransferToDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  int64_t total_size = 0;
  for (auto& tensor : tensors) {
    xla::ifrt::Device* ifrt_device = StringToIfrtDevice(tensor->device());

    total_size += xla::ShapeUtil::ByteSizeOf(tensor->shape());

    tsl::RCReference<xla::ifrt::Array> buffer =
        client_
            ->MakeArrayFromHostBuffer(
                tensor->data(),
                xla::ifrt::ToDType(tensor->primitive_type()).value(),
                xla::ifrt::Shape(tensor->dimensions()), tensor->byte_strides(),
                // TODO: what is MemoryKind?
                xla::ifrt::SingleDeviceSharding::Create(
                    ifrt_device, xla::ifrt::MemoryKind()),
                xla::ifrt::Client::HostBufferSemantics::
                    kImmutableUntilTransferCompletes,
                [tensor, timed]() { /* frees tensor and timer */ })
            .value();

    ComputationClient::DataPtr data =
        std::make_shared<IfrtData>(tensor->device(), tensor->shape(), buffer);
    datas.push_back(data);
  }
  OutboundDataMetric()->AddSample(total_size);
  CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr IfrtComputationClient::TransferShardsToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
    std::string device, xla::Shape shape, xla::OpSharding sharding) {
  tsl::profiler::TraceMe activity(
      "IfrtComputationClient::TransferShardsToDevice",
      tsl::profiler::TraceMeLevel::kInfo);
  // TODO(jonbolin): Consider using CopyToDevice when sharding is REPLICATED.
  // We are opting out of CopyToDevice for now due to the synchronization
  // issues observed in ShardingUtil::InputHandler, but because CopyToDevice
  // directly copies buffers between devices using ICI, it can be much faster
  // than transferring from the host to each device.
  auto data_shards = TransferToDevice(tensor_shards);
  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  std::vector<xla::ifrt::Shape> shard_shapes;
  for (auto& shard : data_shards) {
    auto ifrt_shard = std::dynamic_pointer_cast<IfrtData>(shard);
    arrays.push_back(ifrt_shard->buffer);
    shard_shapes.push_back(ifrt_shard->buffer->shape());
  }
  xla::ifrt::Shape ifrt_shape(shape.dimensions());
  tsl::RCReference<xla::ifrt::DeviceList> devices_list =
      xla::ifrt::BasicDeviceList::Create(
          {client_->addressable_devices().begin(),
           client_->addressable_devices().end()});
  std::unique_ptr<xla::ifrt::Sharding> ifrt_sharding =
      xla::ifrt::ConcreteSharding::Create(devices_list, xla::ifrt::MemoryKind(),
                                          ifrt_shape, shard_shapes);
  // TODO: Attach HloSharding instead when it is supported
  // std::unique_ptr<xla::ifrt::Sharding> ifrt_sharding =
  // xla::ifrt::HloSharding::Create(
  //   devices_list,
  //   xla::ifrt::MemoryKind(),
  //   xla::HloSharding::FromProto(sharding).value()
  // );
  tsl::RCReference<xla::ifrt::Array> sharded_array =
      client_
          ->AssembleArrayFromSingleDeviceArrays(
              ifrt_shape, std::move(ifrt_sharding), absl::MakeSpan(arrays),
              xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
          .value();
  return std::make_shared<IfrtData>(device, shape, sharded_array, sharding);
}

ComputationClient::DataPtr IfrtComputationClient::CopyToDevice(
    ComputationClient::DataPtr data, std::string dst) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

tsl::RCReference<xla::ifrt::Array> IfrtComputationClient::ReplicateShardedData(
    const std::shared_ptr<IfrtData> handle) {
  if (handle->buffer->sharding().devices()->size() == 1) {
    return handle->buffer;
  }

  XLA_COUNTER("ReplicateShardedData", 1);
  TF_VLOG(1) << "ReplicateShardedData (handle=" << handle->GetHandle()
             << ", shape=" << handle->shape() << ")";
  // TODO: handle replicated data
  xla::XlaBuilder builder("ReplicateShardedData");
  xla::Shape shape = handle->shape();
  builder.SetSharding(handle->GetSharding());

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
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());

  std::string device = GetDefaultDevice();
  std::vector<torch_xla::runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), device,
                       GetCompilationDevices(device, {}), &shape,
                       /*should_wrap_parameter=*/false,
                       /*is_sharded=*/true,
                       /*allow_spmd_sharding_propagation_to_output=*/false});
  std::vector<
      std::shared_ptr<torch_xla::runtime::ComputationClient::Computation>>
      computations = Compile(std::move(instances));

  XLA_CHECK_EQ(handle->buffer->sharding().devices()->size(),
               GetLocalDevices().size());

  torch_xla::runtime::ComputationClient::ExecuteReplicatedOptions
      execute_options;

  auto sharded_results = ExecuteReplicated(*computations.front(), {{handle}},
                                           GetLocalDevices(), execute_options);
  auto replicated_output =
      std::dynamic_pointer_cast<IfrtData>(sharded_results[0])
          ->buffer->FullyReplicatedShard(
              xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
  // TODO: sanity check outputs
  return *replicated_output;
}

std::uintptr_t IfrtComputationClient::UnsafeBufferPointer(
    const DataPtr handle) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::shared_ptr<xla::PjRtBuffer> IfrtComputationClient::GetPjRtBuffer(
    const DataPtr handle) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::vector<xla::Literal> IfrtComputationClient::TransferFromDevice(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromDeviceMetric());
  tsl::profiler::TraceMe activity("IfrtComputationClient::TransferFromDevice",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  int64_t total_size = 0;
  for (auto handle : handles) {
    // Use XLA replication to reassemble the sharded data. If input handle
    // is not sharded, then it is a no-op.
    auto ifrt_data = std::dynamic_pointer_cast<IfrtData>(handle);
    tsl::RCReference<xla::ifrt::Array> replicated_array =
        ReplicateShardedData(ifrt_data);

    // TODO: handle dynamic shapes
    auto& literal = literals.emplace_back(
        xla::ShapeUtil::DeviceShapeToHostShape(ifrt_data->shape()));
    std::vector<int64_t> byte_strides(literal.shape().dimensions_size());
    XLA_CHECK_OK(xla::ShapeUtil::ByteStrides(literal.shape(),
                                             absl::MakeSpan(byte_strides)));
    XLA_CHECK_OK(
        replicated_array
            ->CopyToHostBuffer(literal.untyped_data(), byte_strides,
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());

    total_size += literal.size_bytes();
  }
  InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> IfrtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  tsl::profiler::TraceMe activity("IfrtComputationClient::Compile",
                                  tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::ComputationPtr> computations;

  for (auto& instance : instances) {
    xla::CompileOptions compile_options;
    if (instance.is_sharded) {
      // TODO(yeounoh) multi-host, multi-slice configurations
      compile_options.executable_build_options.set_use_spmd_partitioning(true);
      // We can override the compiler's default behavior to replicate the
      // outputs.
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
      XLA_ERROR() << "Only SPMD compilation is supported";
    }

    // Convert HLO to StableHLO for Ifrt client compilation.
    mlir::MLIRContext context;
    mlir::ModuleOp mlir_module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    torch_xla::ConvertHloToStableHlo(instance.computation.mutable_proto(),
                                     &mlir_module);
    std::unique_ptr<xla::ifrt::LoadedExecutable> executable =
        ConsumeValue(client_->GetDefaultCompiler()->Compile(
            std::make_unique<xla::ifrt::HloProgram>(std::move(mlir_module)),
            std::make_unique<xla::ifrt::XlaCompileOptions>(compile_options)));
    StableHloCompileCounter()->AddValue(1);

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());

    std::shared_ptr<IfrtComputation> ifrt_computation =
        std::make_shared<IfrtComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            instance.devices, std::move(executable));

    computations.push_back(ifrt_computation);

    CreateCompileHandlesCounter()->AddValue(1);
  }

  return computations;
}

std::vector<ComputationClient::DataPtr>
IfrtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  // TODO: Implement sharded exec in IFRT
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::vector<ComputationClient::DataPtr>
IfrtComputationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    const absl::Span<const ComputationClient::DataPtr> arguments,
    // TODO: devices isn't doing anything helpful here
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteReplicated` and the async work in `Execute` are
  // complete; a copy is held from the lambda that releases it when done.
  auto timed =
      std::make_shared<metrics::TimedSection>(ExecuteReplicatedMetric());
  tsl::profiler::TraceMe activity("IfrtComputationClient::ExecuteReplicated",
                                  tsl::profiler::TraceMeLevel::kInfo);
  const IfrtComputation& ifrt_computation =
      dynamic_cast<const IfrtComputation&>(computation);

  std::vector<tsl::RCReference<xla::ifrt::Array>> argument_handles(
      arguments.size());
  {
    absl::BlockingCounter counter(arguments.size());

    // Cost to handle one input argument. See tsl::ThreadPool::ParallelFor
    // documentation
    static const int32_t argument_handle_cost_ns = 1000;
    pool_.ParallelFor(arguments.size(), argument_handle_cost_ns,
                      [&](int64_t start, int64_t end) {
                        for (int32_t i = start; i < end; ++i) {
                          auto ifrt_data =
                              std::dynamic_pointer_cast<IfrtData>(arguments[i]);
                          argument_handles[i] = ifrt_data->buffer;
                          counter.DecrementCount();
                        }
                      });
    counter.Wait();
  }

  xla::ifrt::ExecuteOptions execute_options;

  TF_VLOG(5) << "ExecuteReplicated acquiring IFRT device lock for "
             << spmd_device_str;
  auto op_tracker = operation_manager_.StartOperation(spmd_device_str);
  TF_VLOG(5) << "ExecuteReplicated acquiring IFRT device lock for "
             << spmd_device_str << " Done";

  xla::ifrt::LoadedExecutable::ExecuteResult result =
      ifrt_computation.executable
          ->Execute(absl::MakeSpan(argument_handles), execute_options,
                    std::nullopt)
          .value();

  result.status.OnReady(std::move([timed, op_tracker = std::move(op_tracker)](
                                      absl::Status status) mutable {
    timed.reset();
    TF_VLOG(3)
        << "ExecuteReplicated returned_future->OnReady finished with status "
        << status;
  }));

  auto outputs = result.outputs;

  const std::vector<xla::OpSharding>& output_shardings =
      ifrt_computation.output_shardings_
          ? *ifrt_computation.output_shardings_
          : std::vector(outputs.size(),
                        xla::HloSharding::Replicate().ToProto());
  XLA_CHECK_EQ(output_shardings.size(), outputs.size());

  std::vector<ComputationClient::DataPtr> data_handles(outputs.size());
  {
    absl::BlockingCounter counter(outputs.size());

    // Cost to handle one output. See tsl::ThreadPool::ParallelFor
    // documentation.
    static const int32_t result_handle_cost_ns = 2000;
    pool_.ParallelFor(outputs.size(), result_handle_cost_ns,
                      [&](int64_t start, int64_t end) {
                        for (int32_t i = start; i < end; ++i) {
                          data_handles[i] = std::make_shared<IfrtData>(
                              spmd_device_str, outputs[i], output_shardings[i]);
                          counter.DecrementCount();
                        }
                      });
    counter.Wait();
  }

  TF_VLOG(1) << "Returning " << data_handles.size() << " sharded outputs.";
  return data_handles;
}

size_t IfrtComputationClient::GetNumDevices() const {
  return client_->addressable_device_count();
}

std::string IfrtComputationClient::GetDefaultDevice() const {
  return IfrtDeviceToString(client_->addressable_devices()[0]);
}

std::vector<std::string> IfrtComputationClient::GetLocalDevices() const {
  return IfrtDevicesToString(client_->addressable_devices());
}

std::vector<std::string> IfrtComputationClient::GetAllDevices() const {
  return IfrtDevicesToString(client_->devices());
}

int IfrtComputationClient::GetNumProcesses() const {
  int max_process_index = client_->process_index();
  for (auto* device : client_->devices()) {
    max_process_index = std::max(max_process_index, device->ProcessIndex());
  }

  return max_process_index + 1;
};

std::string IfrtComputationClient::GetDeviceKind(const std::string& device) {
  return std::string(StringToIfrtDevice(device)->Kind());
}

const absl::flat_hash_map<
    std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>
IfrtComputationClient::GetDeviceAttributes(const std::string& device) {
  return xla::ifrt::ToPjRtAttributeMap(
      IfrtComputationClient::StringToIfrtDevice(device)->Attributes());
}

void IfrtComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
IfrtComputationClient::GetReplicationDevices() {
  return replication_devices_;
}

xla::ifrt::Device* IfrtComputationClient::StringToIfrtDevice(
    const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
      << "Unknown device " << device;
  xla::ifrt::Device* ifrt_device = string_to_device_[device];
  return ifrt_device;
}

void IfrtComputationClient::WaitDeviceOps(
    absl::Span<const std::string> devices) {
  TF_VLOG(3) << "Waiting for " << absl::StrJoin(devices, ", ");
  operation_manager_.WaitForDevices(devices.empty() ? GetLocalDevices()
                                                    : devices);
}

std::map<std::string, Metric> IfrtComputationClient::GetMetrics() const {
  // TODO(jonbolin): Add any Ifrt-client-specific metrics here
  return {};
}

}  // namespace runtime
}  // namespace torch_xla
