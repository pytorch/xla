#include "tensorflow/compiler/xla/xla_client/xla_computation_client.h"

#include "grpc++/create_channel.h"
#include "grpc++/support/channel_arguments.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"

namespace xla {

XlaComputationClient::XlaComputationClient(
    XlaComputationClient::Options options)
    : options_(std::move(options)) {
  int64 device_count = 1;
  if (!options_.host_name.empty()) {
    ::grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    auto channel = ::grpc::CreateCustomChannel(
        absl::StrCat(options_.host_name, ":", options_.port),
        ::grpc::InsecureChannelCredentials(), ch_args);
    channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN)));
    TF_LOG(INFO) << "Channel to '" << options_.host_name
                 << "' is connected on port " << options_.port;

    xla_service_ = grpc::XlaService::NewStub(channel);
    stub_.reset(new GRPCStub(xla_service_.get()));
    client_ptr_.reset(new Client(stub_.get()));
    client_ = client_ptr_.get();
    device_count = options_.platform == "TPU" ? 8 : 1;
  } else {
    se::Platform* platform = nullptr;
    if (!options_.platform.empty()) {
      platform = PlatformUtil::GetPlatform(options_.platform).ValueOrDie();
    }
    TF_LOG(INFO) << "Creating XLA computation client for '"
                 << (options_.platform.empty() ? "default" : options_.platform)
                 << "' platform";
    LocalClient* local_client =
        ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();
    device_count = local_client->device_count();
    client_ = local_client;
  }
  device_handles_ = client_->GetDeviceHandles(device_count).ValueOrDie();
  StartHandleReleaser();
}

void XlaComputationClient::FlushLazyReleases() {
  // Activate the lazy handle releaser and wait for it to complete our run.
  size_t run_id = triggered_task_->Activate();
  triggered_task_->WaitForRun(run_id);
}

size_t XlaComputationClient::ForceReleaseHandles(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) {
  size_t released = 0;
  for (auto& handle : handles) {
    XlaData* xla_data = dynamic_cast<XlaData*>(handle.get());
    if (ReleaseXlaData(xla_data)) {
      ++released;
    }
  }
  return released;
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XlaComputationClient::TransferToServer(
    tensorflow::gtl::ArraySlice<const LiteralDevice> literals) {
  metrics::TimedSection timed(TransferToServerMetric());

  // This can be made parallel, WRT literal creation.
  int64 total_size = 0;
  std::vector<std::shared_ptr<Data>> results;
  for (auto& literal_device : literals) {
    string device = GetEffectiveDevice(literal_device.device);
    Literal literal_storage;
    const Literal& literal = literal_device.GetLiteral(&literal_storage);
    std::unique_ptr<GlobalData> handle =
        client_->TransferToServer(literal, &GetDeviceHandle(device))
            .ValueOrDie();
    results.push_back(std::make_shared<XlaData>(
        std::move(handle), device, literal.shape(),
        [this](XlaData* xla_data) { ReleaseXlaData(xla_data); }));
    total_size += literal.size_bytes();
  }
  CreateHandlesCounter()->AddValue(literals.size());
  OutboundDataMetric()->AddSample(total_size);
  return results;
}

std::vector<Literal> XlaComputationClient::TransferFromServer(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());

  int64 total_size = 0;
  std::vector<Literal> results;
  for (auto& handle : handles) {
    const XlaData& xla_data = dynamic_cast<const XlaData&>(*handle);
    results.push_back(
        client_->Transfer(*xla_data.handle, /*shape_with_layout=*/nullptr)
            .ValueOrDie());
    total_size += results.back().size_bytes();
  }
  InboundDataMetric()->AddSample(total_size);
  return results;
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XlaComputationClient::ExecuteComputation(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
    const ExecuteComputationOptions& options) {
  metrics::TimedSection timed(ExecuteMetric());

  std::string effective_device = GetEffectiveDevice(device);
  std::vector<GlobalData*> arguments_data =
      GetArgumentsData(arguments, effective_device);
  ExecutionOptions eo;
  *eo.mutable_debug_options() = GetDebugOptionsFromFlags();
  *eo.add_device_handles() = GetDeviceHandle(effective_device);
  if (options.output_shape != nullptr) {
    *eo.mutable_shape_with_output_layout() = options.output_shape->ToProto();
  }
  StatusOr<std::unique_ptr<GlobalData>> result_or_status =
      client_->Execute(computation, arguments_data, &eo);
  xrt_util::CheckComputationStatus(result_or_status.status(), {&computation});

  const Shape* output_shape = options.output_shape;
  ProgramShape program_shape;
  if (output_shape == nullptr) {
    program_shape = computation.GetProgramShape().ValueOrDie();
    output_shape = &program_shape.result();
  }

  std::shared_ptr<Data> data = std::make_shared<XlaData>(
      std::move(result_or_status.ValueOrDie()), effective_device, *output_shape,
      [this](XlaData* xla_data) { ReleaseXlaData(xla_data); });
  CreateHandlesCounter()->AddValue(1);
  std::vector<std::shared_ptr<Data>> results;
  if (ShapeUtil::IsTuple(*output_shape) && options.explode_tuple) {
    auto tuple_results = DeconstructTuple({data});
    results = std::move(tuple_results.front());
  } else {
    results.push_back(std::move(data));
  }
  return results;
}

std::vector<std::vector<std::shared_ptr<ComputationClient::Data>>>
XlaComputationClient::ExecuteReplicated(
    const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    const ExecuteReplicatedOptions& options) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());
  TF_LOG(FATAL) << "ExecuteReplicated() API not yet implemented!";
  std::vector<std::vector<std::shared_ptr<Data>>> results;
  return results;
}

std::vector<std::vector<std::shared_ptr<ComputationClient::Data>>>
XlaComputationClient::ExecuteParallel(
    tensorflow::gtl::ArraySlice<const XlaComputation> computations,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    const ExecuteParallelOptions& options) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  std::vector<const XlaComputation*> computations_pointers;
  std::vector<Client::XlaComputationInstance> instances;
  for (size_t i = 0; i < computations.size(); ++i) {
    std::string effective_device = GetEffectiveDevice(devices[i]);
    std::vector<GlobalData*> arguments_data =
        GetArgumentsData(arguments[i], effective_device);
    ExecutionOptions eo;
    *eo.mutable_debug_options() = GetDebugOptionsFromFlags();
    *eo.add_device_handles() = GetDeviceHandle(effective_device);
    if (i < options.output_shapes.size() &&
        options.output_shapes[i] != nullptr) {
      *eo.mutable_shape_with_output_layout() =
          options.output_shapes[i]->ToProto();
    }
    instances.emplace_back(computations[i], std::move(arguments_data),
                           std::move(eo), nullptr);
    computations_pointers.push_back(&computations[i]);
  }

  StatusOr<std::vector<std::unique_ptr<GlobalData>>> results_or_status =
      client_->ExecuteParallel(instances);
  xrt_util::CheckComputationStatus(results_or_status.status(),
                                   computations_pointers);
  std::vector<std::unique_ptr<GlobalData>> exec_results(
      std::move(results_or_status.ValueOrDie()));
  XLA_CHECK_EQ(exec_results.size(), computations.size());
  std::vector<std::vector<std::shared_ptr<Data>>> results;
  for (size_t i = 0; i < computations.size(); ++i) {
    ProgramShape program_shape;
    const Shape* output_shape =
        (i < options.output_shapes.size()) ? options.output_shapes[i] : nullptr;
    if (output_shape == nullptr) {
      program_shape = computations[i].GetProgramShape().ValueOrDie();
      output_shape = &program_shape.result();
    }
    std::shared_ptr<Data> data = std::make_shared<XlaData>(
        std::move(exec_results[i]), GetEffectiveDevice(devices[i]),
        *output_shape, [this](XlaData* xla_data) { ReleaseXlaData(xla_data); });
    std::vector<std::shared_ptr<Data>> computation_results;
    if (ShapeUtil::IsTuple(*output_shape) && options.explode_tuple) {
      auto tuple_results = DeconstructTuple({data});
      computation_results = std::move(tuple_results.front());
    } else {
      computation_results.push_back(std::move(data));
    }
    results.push_back(std::move(computation_results));
  }
  CreateHandlesCounter()->AddValue(computations.size());
  return results;
}

std::vector<std::vector<std::shared_ptr<ComputationClient::Data>>>
XlaComputationClient::DeconstructTuple(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) {
  metrics::TimedSection timed(DeconstructTupleMetric());
  std::vector<std::vector<std::shared_ptr<Data>>> results;
  for (auto& tuple : tuples) {
    const XlaData& xla_data = dynamic_cast<const XlaData&>(*tuple);
    auto exploded_tuple =
        client_->DeconstructTuple(*xla_data.handle).ValueOrDie();
    std::vector<std::shared_ptr<Data>> tuple_results;
    for (int64 i = 0; i < exploded_tuple.size(); ++i) {
      tuple_results.push_back(std::make_shared<XlaData>(
          std::move(exploded_tuple[i]), xla_data.device(),
          ShapeUtil::GetTupleElementShape(xla_data.shape(), i),
          [this](XlaData* xla_data) { ReleaseXlaData(xla_data); }));
    }
    results.push_back(std::move(tuple_results));
    CreateHandlesCounter()->AddValue(exploded_tuple.size());
  }
  return results;
}

std::vector<GlobalData*> XlaComputationClient::GetArgumentsData(
    tensorflow::gtl::ArraySlice<Data*> arguments, const string& device) const {
  std::vector<GlobalData*> arguments_data;
  for (auto data : arguments) {
    XlaData* xla_data = dynamic_cast<XlaData*>(data);
    XLA_CHECK_EQ(xla_data->device(), device);
    arguments_data.push_back(xla_data->handle.get());
  }
  return arguments_data;
}

const DeviceHandle& XlaComputationClient::GetDeviceHandle(
    const string& device) const {
  int64 ordinal = GetDeviceOrdinal(device);
  XLA_CHECK_LT(ordinal, device_handles_.size()) << device;
  return device_handles_[ordinal];
}

string XlaComputationClient::GetEffectiveDevice(const string& device) const {
  if (device.empty()) {
    return GetDefaultDevice();
  }
  if (device[0] == ':') {
    // Allow devices with ordinal only specification, to expand from the default
    // device type.
    return options_.platform + device;
  }
  return device;
}

bool XlaComputationClient::ReleaseXlaData(XlaData* xla_data) {
  bool released = false;
  {
    std::lock_guard<std::mutex> lock(lock_);
    std::unique_ptr<GlobalData> handle = xla_data->Release();
    if (handle != nullptr) {
      released_handles_.push_back(std::move(handle));
      released = true;
    }
  }
  if (released) {
    triggered_task_->Activate();
    ReleaseHandlesCounter()->AddValue(1);
  }
  return released;
}

void XlaComputationClient::StartHandleReleaser() {
  triggered_task_.reset(
      new xla_util::TriggeredTask([this]() { HandleReleaser(); }));
}

void XlaComputationClient::HandleReleaser() {
  std::vector<std::unique_ptr<GlobalData>> released_handles;
  {
    std::lock_guard<std::mutex> lock(lock_);
    released_handles.swap(released_handles_);
  }
  if (!released_handles.empty()) {
    size_t num_handles = released_handles.size();
    metrics::TimedSection timed(ReleaseHandlesTimeMetric());
    GlobalData::Release(std::move(released_handles));
    DestroyHandlesCounter()->AddValue(num_handles);
  }
}

string XlaComputationClient::GetDefaultDevice() const {
  return options_.platform + ":0";
}

}  // namespace xla
