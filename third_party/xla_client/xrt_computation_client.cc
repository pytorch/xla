#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include <functional>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {
namespace {

static const char* const kCpuDevice = "/device:CPU:0";

}  // namespace

XrtComputationClient::XrtComputationClient(
    XrtComputationClient::Options options)
    : options_(std::move(options)) {
  auto default_device_target =
      options_.device_map.find(options_.default_device);
  CHECK(default_device_target != options_.device_map.end());
  for (const auto& dev_target : options_.device_map) {
    LOG(INFO) << "XRT device " << dev_target.first << " -> "
              << dev_target.second;
  }
  LOG(INFO) << "XRT default device: " << default_device_target->first;
  InitializeDevices();
}

std::shared_ptr<ComputationClient::Data>
XrtComputationClient::ExecuteComputation(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteMetric());
  ApiCallInitialize();

  std::vector<string> devices;
  tensorflow::ClientSession::FeedType feed_inputs;
  auto exec_ops =
      CreateExecuteOps(computation, BuildParallelArguments(arguments),
                       output_shape, &devices, &feed_inputs);
  SessionData* session = GetSessionForDevice(devices.front());
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root.status());
  xrt_util::CheckComputationStatus(
      session->session.Run(feed_inputs, {exec_ops.front().execute_output},
                           &outputs),
      computation);
  CHECK_EQ(outputs.size(), 1);

  return std::make_shared<XrtData>(
      devices.front(), outputs[0].scalar<int64>()(),
      exec_ops.front().result_shape,
      [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
}

std::unique_ptr<Literal> XrtComputationClient::ExecuteComputationAndTransfer(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteTrfMetric());
  ApiCallInitialize();

  ProgramShape program_shape;
  if (output_shape == nullptr) {
    program_shape = computation.GetProgramShape().ValueOrDie();
    output_shape = &program_shape.result();
  }
  string device = GetArgumentsDevice(arguments);
  auto xrt_computation = CreateXrtComputation(computation, /*num_replicas=*/1,
                                              {device}, output_shape);
  tensorflow::ClientSession::FeedType feed_inputs;
  auto inputs = GetArgumentsInputs(arguments, device, &feed_inputs);
  SessionData* session = GetSessionForDevice(device);
  tensorflow::Scope device_scope =
      session->root.WithDevice(TorchDeviceToXrtDevice(device));
  const CachedNode& cached_node =
      GetCompileExecuteReadNode(device_scope, device);

  feed_inputs.insert(
      {cached_node.holders[0], xrt_computation->SerializeAsString()});

  xrt::XRTExecutionConfig exec_config;
  exec_config.set_release_input_handles(false);
  exec_config.set_release_compilation_handle(true);
  feed_inputs.insert({cached_node.holders[1], exec_config.SerializeAsString()});
  feed_inputs.insert({cached_node.holders[2], inputs});

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root.status());
  xrt_util::CheckComputationStatus(
      session->session.Run(feed_inputs, {*cached_node.output}, &outputs),
      computation);
  CHECK_EQ(outputs.size(), 1);

  LiteralProto response;
  CHECK(response.ParseFromString(outputs[0].scalar<string>()()));
  std::unique_ptr<Literal> result(
      new Literal(Literal::CreateFromProto(response).ValueOrDie()));
  InboundDataMetric()->AddSample(result->size_bytes());
  return result;
}

std::shared_ptr<ComputationClient::Data>
XrtComputationClient::TransferParameterToServer(const Literal& literal,
                                                const string& device) {
  metrics::TimedSection timed(TransferMetric());
  OutboundDataMetric()->AddSample(literal.size_bytes());
  ApiCallInitialize();

  string effective_device = GetEffectiveDevice(device);
  const string& xrt_device = TorchDeviceToXrtDevice(effective_device);
  SessionData* session = GetSessionForXrtDevice(xrt_device);
  xrt::XLAAllocation alloc;
  alloc.set_device_ordinal(GetDeviceOrdinal(xrt_device));
  *alloc.mutable_value() = literal.ToProto();

  tensorflow::ClientSession::FeedType feed_inputs;
  tensorflow::Scope device_scope = session->root.WithDevice(xrt_device);
  const CachedNode& cached_node =
      GetAllocateNode(device_scope, effective_device);
  feed_inputs.insert({cached_node.holders[0], alloc.SerializeAsString()});

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root.status());
  TF_CHECK_OK(
      session->session.Run(feed_inputs, {*cached_node.output}, &outputs));
  CHECK_EQ(outputs.size(), 1);
  return std::make_shared<XrtData>(
      effective_device, outputs[0].scalar<int64>()(), literal.shape(),
      [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XrtComputationClient::ExecuteReplicated(
    const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments,
    const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteReplMetric());
  ApiCallInitialize();

  std::vector<string> devices;
  tensorflow::ClientSession::FeedType feed_inputs;
  auto exec_ops = CreateExecuteOps(computation, arguments, output_shape,
                                   &devices, &feed_inputs);
  // In the PyTorch/XRT interface we keep a map (options_.workers_map) from a
  // worker+taskno, to the GRPC server which is the entry point for that worker.
  // Since XRT could re-distribute ops internally, if we have N hosts
  // (worker+taskno), we could have all the workers pointing to a single GRPC
  // entry point, or we could have each worker pointing directly to the target
  // host.
  // The advantage of the latter approach, is that we do not bottleneck
  // (especially when feeding inputs) the single GRPC entry point.
  // Using the N:1 approach, the session_replicas below will contain a single
  // session, and all the replica executions will go through it (and distributed
  // by XRT on the service side).
  // Chosing the 1:1 approach (one session per worker), we will have N sessions
  // within the session_replicas map, which we will be executing independently.
  std::map<SessionData*, std::vector<size_t>> session_replicas;
  for (size_t i = 0; i < devices.size(); ++i) {
    SessionData* session = GetSessionForDevice(devices[i]);
    session_replicas[session].push_back(i);
  }
  // TODO(dlibenzi): These could be run in parallel.
  std::vector<std::shared_ptr<Data>> results(devices.size());
  for (auto& sess_replica : session_replicas) {
    std::vector<tensorflow::Output> exec_nodes;
    for (auto replica : sess_replica.second) {
      exec_nodes.push_back(exec_ops[replica].execute_output);
    }
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(sess_replica.first->root.status());
    xrt_util::CheckComputationStatus(
        sess_replica.first->session.Run(feed_inputs, exec_nodes, &outputs),
        computation);
    CHECK_EQ(outputs.size(), exec_nodes.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto replica = sess_replica.second[i];
      results[replica] = std::make_shared<XrtData>(
          devices[replica], outputs[i].scalar<int64>()(),
          exec_ops[replica].result_shape,
          [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
    }
  }
  return results;
}

StatusOr<std::vector<std::shared_ptr<ComputationClient::Data>>>
XrtComputationClient::DeconstructTuple(const Data& data) {
  metrics::TimedSection timed(DeconstructTupleMetric());
  ApiCallInitialize();

  const XrtData& xrt_data = dynamic_cast<const XrtData&>(data);
  SessionData* session = GetSessionForDevice(xrt_data.device());
  tensorflow::Scope device_scope =
      session->root.WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
  tensorflow::Scope cpu_scope = session->root.WithDevice(kCpuDevice);
  int64 count = ShapeUtil::TupleElementCount(xrt_data.shape());
  std::vector<tensorflow::Output> sub_outputs;
  tensorflow::ClientSession::FeedType feed_inputs;
  for (int64 i = 0; i < count; ++i) {
    const CachedNode& cached_node =
        GetSubTupleNode(device_scope, xrt_data.device());
    feed_inputs.insert({cached_node.holders[0], xrt_data.handle});
    tensorflow::Tensor index_tensor(tensorflow::DT_INT32,
                                    tensorflow::TensorShape({1}));
    index_tensor.flat<tensorflow::int32>()(0) = i;
    feed_inputs.insert({cached_node.holders[1], index_tensor});
    sub_outputs.push_back(*cached_node.output);
  }
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root.status());
  TF_CHECK_OK(session->session.Run(feed_inputs, sub_outputs, &outputs));
  CHECK_EQ(outputs.size(), count);

  std::vector<std::shared_ptr<Data>> components;
  for (int64 i = 0; i < count; ++i) {
    components.push_back(std::make_shared<XrtData>(
        xrt_data.device(), outputs[i].scalar<int64>()(),
        ShapeUtil::GetTupleElementShape(xrt_data.shape(), i),
        [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); }));
  }
  return std::move(components);
}

XrtComputationClient::SessionData* XrtComputationClient::GetSessionForTarget(
    const string& target) {
  auto target_session = session_map_.find(target);
  if (target_session == session_map_.end()) {
    target_session =
        session_map_
            .emplace(target,
                     std::unique_ptr<SessionData>(new SessionData(target)))
            .first;
  }
  return target_session->second.get();
}

XrtComputationClient::SessionData* XrtComputationClient::GetSessionForXrtDevice(
    const string& xrt_device) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  return GetSessionForTarget(worker_hostport.second);
}

XrtComputationClient::SessionData* XrtComputationClient::GetSessionForDevice(
    const string& device) {
  return GetSessionForXrtDevice(TorchDeviceToXrtDevice(device));
}

const string& XrtComputationClient::GetEffectiveDevice(
    const string& device) const {
  return !device.empty() ? device : options_.default_device;
}

const string& XrtComputationClient::TorchDeviceToXrtDevice(
    const string& device) const {
  auto device_target = options_.device_map.find(GetEffectiveDevice(device));
  CHECK(device_target != options_.device_map.end())
      << "Unable to find device: " << device;
  return device_target->second;
}

std::unique_ptr<xrt::XLAComputation> XrtComputationClient::CreateXrtComputation(
    const XlaComputation& computation, int64 num_replicas,
    const std::vector<string>& devices, const Shape* output_shape) const {
  CHECK_EQ(num_replicas, devices.size());
  std::unique_ptr<xrt::XLAComputation> xrt_computation(
      new xrt::XLAComputation());
  auto config = xrt_computation->mutable_config();
  config->set_num_replicas(num_replicas);
  config->set_num_cores_per_replica(1);
  if (num_replicas > 1) {
    auto device_assignment = config->mutable_device_assignment();
    auto computation_device = device_assignment->add_computation_devices();
    for (int64 i = 0; i < num_replicas; ++i) {
      const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
      const auto& core_coords = GetDeviceMeshCoords(xrt_device);
      auto replica_device = computation_device->add_replica_devices();
      for (auto coord : core_coords) {
        replica_device->add_value(coord);
      }
    }
  }
  *config->mutable_program_shape() = computation.GetProgramShape().ValueOrDie();
  if (output_shape != nullptr) {
    *config->mutable_program_shape()->mutable_result() = *output_shape;
  }
  *xrt_computation->mutable_hlo_snapshot() =
      *computation.Snapshot().ValueOrDie();
  return xrt_computation;
}

string XrtComputationClient::GetArgumentsDevice(
    tensorflow::gtl::ArraySlice<Data*> arguments) const {
  xla_util::Unique<string> unique_device;
  for (size_t i = 0; i < arguments.size(); ++i) {
    XrtData* xrt_data = dynamic_cast<XrtData*>(arguments[i]);
    unique_device.set(xrt_data->device());
  }
  // If the computation has no arguments, use the default device.
  // Maybe the execute-computation APIs needs to be more explicit about it.
  return unique_device ? *unique_device : options_.default_device;
}

std::vector<string> XrtComputationClient::GetReplicasDevices(
    const std::vector<std::vector<Data*>>& arguments) const {
  std::vector<string> devices;
  std::set<string> unique_devices;
  for (size_t i = 0; i < arguments.size(); ++i) {
    devices.push_back(GetArgumentsDevice(arguments[i]));
    CHECK(unique_devices.insert(devices.back()).second)
        << "Cannot have two different replicas using the same device: "
        << devices.back();
  }
  return devices;
}

tensorflow::Tensor XrtComputationClient::GetArgumentsInputs(
    tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  tensorflow::Tensor inputs_tensor(tensorflow::DT_INT64,
                                   tensorflow::TensorShape({arguments.size()}));
  for (size_t i = 0; i < arguments.size(); ++i) {
    XrtData* xrt_data = dynamic_cast<XrtData*>(arguments[i]);
    CHECK_EQ(device, xrt_data->device());
    inputs_tensor.flat<tensorflow::int64>()(i) = xrt_data->handle;
  }
  return inputs_tensor;
}

std::vector<XrtComputationClient::ExecuteContext>
XrtComputationClient::CreateExecuteOps(
    const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments, const Shape* output_shape,
    std::vector<string>* devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  ProgramShape program_shape;
  if (output_shape == nullptr) {
    program_shape = computation.GetProgramShape().ValueOrDie();
    output_shape = &program_shape.result();
  }
  *devices = GetReplicasDevices(arguments);
  auto xrt_computation = CreateXrtComputation(computation, arguments.size(),
                                              *devices, output_shape);

  absl::optional<tensorflow::ops::Placeholder> computation_holder;
  xla_util::Unique<SessionData*> unique_session;
  std::vector<ExecuteContext> exec_ops;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto inputs = GetArgumentsInputs(arguments[i], devices->at(i), feed_inputs);
    const string& xrt_device = TorchDeviceToXrtDevice(devices->at(i));
    SessionData* session =
        unique_session.set(GetSessionForXrtDevice(xrt_device)).second;
    tensorflow::Scope device_scope = session->root.WithDevice(xrt_device);
    const CachedNode& cached_node =
        GetCompileExecuteNode(device_scope, devices->at(i));
    feed_inputs->insert(
        {cached_node.holders[0], xrt_computation->SerializeAsString()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(true);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.emplace_back(*cached_node.output, *output_shape);
  }
  return exec_ops;
}

void XrtComputationClient::ReleaseHandles(
    tensorflow::gtl::ArraySlice<const DeviceHandle> handles) {
  struct SessionReleases {
    tensorflow::ClientSession::FeedType feed_inputs;
    std::vector<tensorflow::Operation> releases;
  };
  std::map<SessionData*, SessionReleases> session_releases;
  for (auto& handle : handles) {
    SessionData* session = GetSessionForDevice(handle.device);
    SessionReleases* release = &session_releases[session];
    tensorflow::Scope device_scope =
        session->root.WithDevice(TorchDeviceToXrtDevice(handle.device));
    const CachedNode& cached_node =
        GetReleaseAllocationHandleNode(device_scope, handle.device);
    release->feed_inputs.insert({cached_node.holders[0], handle.handle});
    release->releases.push_back(*cached_node.operation);
  }
  for (const auto& session_releases : session_releases) {
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session_releases.first->root.status());
    TF_CHECK_OK(session_releases.first->session.Run(
        session_releases.second.feed_inputs, {},
        session_releases.second.releases, &outputs));
  }
}

void XrtComputationClient::FlushReleasedHandles() {
  ReleaseHandles(released_handles_);
  released_handles_.clear();
}

void XrtComputationClient::ApiCallInitialize() {
  RewindCaches();
  FlushReleasedHandles();
}

void XrtComputationClient::ReleaseXrtData(XrtData* xrt_data) {
  xrt_data->Release();
  released_handles_.emplace_back(xrt_data->device(), xrt_data->handle);
}

std::pair<XrtComputationClient::Worker, string>
XrtComputationClient::GetWorkerForXrtDevice(const string& xrt_device) const {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(xrt_device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task)
      << xrt_device;

  auto worker_hostport =
      options_.workers_map.find(Worker(parsed_device.job, parsed_device.task));
  CHECK(worker_hostport != options_.workers_map.end()) << xrt_device;
  return std::pair<Worker, string>(worker_hostport->first,
                                   worker_hostport->second);
}

const std::vector<int>& XrtComputationClient::GetDeviceMeshCoords(
    const string& xrt_device) const {
  auto it = device_mesh_coords_.find(xrt_device);
  if (it == device_mesh_coords_.end()) {
    LOG(FATAL) << "Missing mesh coordinates for device: " << xrt_device;
  }
  return it->second;
}

tensorflow::tpu::TopologyProto XrtComputationClient::InitializeAndFetchTopology(
    const string& xrt_device) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  LOG(INFO) << "Initializing TPU system for worker "
            << worker_hostport.first.name << ":"
            << worker_hostport.first.task_no << " at "
            << worker_hostport.second;
  string system_device =
      absl::StrCat("/job:", worker_hostport.first.name,
                   "/replica:0/task:", worker_hostport.first.task_no,
                   "/device:TPU_SYSTEM:0");
  SessionData* session = GetSessionForTarget(worker_hostport.second);
  tensorflow::Scope tpu_system_scope = session->root.WithDevice(system_device);
  const auto unique_name =
      tpu_system_scope.GetUniqueNameForOp("ConfigureDistributedTPU");
  auto builder = tensorflow::NodeBuilder(unique_name, "ConfigureDistributedTPU")
                     .Attr("embedding_config", "")
                     .Attr("tpu_embedding_config", "")
                     .Attr("is_global_init", false);
  tpu_system_scope.UpdateBuilder(&builder);

  tensorflow::Node* result;
  session->root.UpdateStatus(
      builder.Finalize(tpu_system_scope.graph(), &result));
  TF_CHECK_OK(tpu_system_scope.status());
  session->root.UpdateStatus(tpu_system_scope.DoShapeInference(result));

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root.status());
  TF_CHECK_OK(session->session.Run({tensorflow::Output(result, 0)}, &outputs));
  CHECK_EQ(outputs.size(), 1);

  tensorflow::tpu::TopologyProto topology_proto;
  CHECK(topology_proto.ParseFromString(outputs[0].scalar<string>()()));
  return topology_proto;
}

void XrtComputationClient::InitializeDevices() {
  auto it = options_.device_map.find("TPU:0");
  if (it != options_.device_map.end()) {
    tensorflow::tpu::TopologyProto topology_proto =
        InitializeAndFetchTopology(it->second);
    LOG(INFO) << "TPU topology: " << topology_proto.DebugString();

    tensorflow::DeviceNameUtils::ParsedName parsed_device;
    CHECK(tensorflow::DeviceNameUtils::ParseFullName(it->second,
                                                     &parsed_device) &&
          parsed_device.has_job)
        << it->second;
    string tpu_job_name = parsed_device.job;
    for (const auto& dev_target : options_.device_map) {
      CHECK(tensorflow::DeviceNameUtils::ParseFullName(dev_target.second,
                                                       &parsed_device) &&
            parsed_device.has_job && parsed_device.has_task &&
            parsed_device.has_id)
          << dev_target.second;
      if (parsed_device.job != tpu_job_name) {
        continue;
      }
      CHECK_LE(parsed_device.task, topology_proto.num_tasks());
      CHECK_LE(parsed_device.id, topology_proto.num_tpu_devices_per_task());
      // The topology proto 'device_coordinates' is a linear list of
      // [num_tasks][devices_per_task][mesh_shape_size] coordinates, where the
      // mesh coordinates are usually [x, y, c] ('x' and 'y' being the spatial
      // chip coordinated and 'c' the core number).
      int64 base_index = parsed_device.task *
                             topology_proto.num_tpu_devices_per_task() *
                             topology_proto.mesh_shape_size() +
                         parsed_device.id * topology_proto.mesh_shape_size();
      std::vector<int> device_mesh_coords(topology_proto.mesh_shape_size());
      for (int i = 0; i < topology_proto.mesh_shape_size(); ++i) {
        device_mesh_coords[i] =
            topology_proto.device_coordinates(base_index + i);
      }
      device_mesh_coords_.insert(
          {dev_target.second, std::move(device_mesh_coords)});
    }
  }
}

string XrtComputationClient::GetDefaultDevice() const {
  return options_.default_device;
}

void XrtComputationClient::RewindCaches() {
  for (auto& key_cache : node_cache_) {
    key_cache.second.rewind();
  }
}

const XrtComputationClient::CachedNode&
XrtComputationClient::GetCompileExecuteNode(const tensorflow::Scope& scope,
                                            const string& device) {
  NodeCache* cache =
      &node_cache_[NodeCacheKey(device, NodeTypes::kCompileExecute)];
  if (cache->empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT64,
             tensorflow::ops::Placeholder::Shape({-1}))});
    auto computation_handle = tensorflow::ops::XRTCompile(scope, holders[0]);
    std::unique_ptr<CachedNode> node(new CachedNode(
        tensorflow::ops::XRTExecute(scope, computation_handle.handle,
                                    holders[1],
                                    {tensorflow::Output(holders[2])}),
        std::move(holders)));
    cache->add(std::move(node));
  }
  return cache->get();
}

const XrtComputationClient::CachedNode&
XrtComputationClient::GetCompileExecuteReadNode(const tensorflow::Scope& scope,
                                                const string& device) {
  NodeCache* cache =
      &node_cache_[NodeCacheKey(device, NodeTypes::kCompileExecuteRead)];
  if (cache->empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT64,
             tensorflow::ops::Placeholder::Shape({-1}))});
    auto computation_handle = tensorflow::ops::XRTCompile(scope, holders[0]);
    auto execute_op = tensorflow::ops::XRTExecute(
        scope, computation_handle.handle, holders[1],
        {tensorflow::Output(holders[2])});
    std::unique_ptr<CachedNode> node(new CachedNode(
        tensorflow::ops::XRTReadLiteralAndRelease(scope, execute_op),
        std::move(holders)));
    cache->add(std::move(node));
  }
  return cache->get();
}

const XrtComputationClient::CachedNode& XrtComputationClient::GetAllocateNode(
    const tensorflow::Scope& scope, const string& device) {
  NodeCache* cache = &node_cache_[NodeCacheKey(device, NodeTypes::kAllocate)];
  if (cache->empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    std::unique_ptr<CachedNode> node(new CachedNode(
        tensorflow::ops::XRTAllocate(scope, holders[0]), std::move(holders)));
    cache->add(std::move(node));
  }
  return cache->get();
}

const XrtComputationClient::CachedNode&
XrtComputationClient::GetReleaseAllocationHandleNode(
    const tensorflow::Scope& scope, const string& device) {
  NodeCache* cache =
      &node_cache_[NodeCacheKey(device, NodeTypes::kReleaseAllocationHandle)];
  if (cache->empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    std::unique_ptr<CachedNode> node(new CachedNode(
        tensorflow::ops::XRTReleaseAllocationHandle(scope, holders[0]),
        std::move(holders)));
    cache->add(std::move(node));
  }
  return cache->get();
}

const XrtComputationClient::CachedNode& XrtComputationClient::GetSubTupleNode(
    const tensorflow::Scope& scope, const string& device) {
  NodeCache* cache = &node_cache_[NodeCacheKey(device, NodeTypes::kSubTuple)];
  if (cache->empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT32,
             tensorflow::ops::Placeholder::Shape({1}))});
    std::unique_ptr<CachedNode> node(new CachedNode(
        tensorflow::ops::XRTSubTuple(scope, holders[0], holders[1]),
        std::move(holders)));
    cache->add(std::move(node));
  }
  return cache->get();
}

std::vector<std::vector<ComputationClient::Data*>>
XrtComputationClient::BuildParallelArguments(
    tensorflow::gtl::ArraySlice<Data*> arguments) {
  std::vector<std::vector<ComputationClient::Data*>> para_arguments(1);
  para_arguments[0].insert(para_arguments[0].end(), arguments.begin(),
                           arguments.end());
  return para_arguments;
}

int64 XrtComputationClient::GetDeviceOrdinal(const string& device) {
  auto pos = device.rfind(':');
  CHECK_NE(pos, string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

}  // namespace xla
