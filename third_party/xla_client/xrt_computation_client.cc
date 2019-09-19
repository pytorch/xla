#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include <cstdlib>
#include <functional>
#include <list>
#include <sstream>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_local_service.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {
namespace {

thread_local std::vector<string> g_replication_devices;

// A simple Tensorflow Allocator which caches Tensor allocations in order to
// avoid paying the kernel's clear_page_c() price.
class TensorAllocator : public tensorflow::Allocator {
  struct AllocKey {
    struct Hash {
      size_t operator()(const AllocKey& hk) const {
        return util::HashCombine(hk.alignment, hk.num_bytes);
      }
    };

    bool operator==(const AllocKey& rhs) const {
      return num_bytes == rhs.num_bytes && alignment == rhs.alignment;
    }

    size_t alignment = 0;
    size_t num_bytes = 0;
  };

  struct AllocBlocks {
    AllocBlocks(const AllocKey& alloc_key) : alloc_key(alloc_key) {}

    AllocKey alloc_key;
    std::vector<void*> blocks;
  };

  using AllocList = std::list<AllocBlocks>;

 public:
  static TensorAllocator* Get() {
    static size_t max_size =
        sys_util::GetEnvInt("XLA_TENSOR_ALLOCATOR_MAXSIZE", 1000000000);
    static TensorAllocator* allocator = new TensorAllocator(max_size);
    return allocator;
  }

  string Name() override { return "XLA_TensorAllocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // We use an alignment-sized area before the memory returned to the caller,
    // to store a pointer to its AllocBlocks.
    alignment = std::max<size_t>(alignment, sizeof(void*));
    // To call aligned_alloc(), num_bytes must be multiple of alignment.
    num_bytes = RoundUpToNearest(num_bytes, alignment);

    AllocKey alloc_key = {alignment, num_bytes};
    void* block = nullptr;
    AllocBlocks* alloc_blocks = nullptr;
    std::lock_guard<std::mutex> lock(lock_);
    auto it = allocs_.find(alloc_key);
    if (it != allocs_.end()) {
      alloc_blocks = &*it->second;
      if (!alloc_blocks->blocks.empty()) {
        block = alloc_blocks->blocks.back();
        alloc_blocks->blocks.pop_back();
      }
      // LRU
      alloc_list_.splice(alloc_list_.begin(), alloc_list_, it->second);
    } else {
      allocs_.emplace(alloc_key,
                      alloc_list_.insert(alloc_list_.begin(), alloc_key));
      alloc_blocks = &alloc_list_.front();
    }
    if (block == nullptr) {
      TrimCache(alloc_key.num_bytes);
      block = NewBlock(alloc_blocks);
    }
    return block;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr != nullptr) {
      // The pointer to AllocBlocks is right before the user memory.
      AllocBlocks* alloc_blocks = reinterpret_cast<AllocBlocks**>(ptr)[-1];
      std::lock_guard<std::mutex> lock(lock_);
      if (alloc_blocks->alloc_key.num_bytes < max_size_) {
        alloc_blocks->blocks.push_back(ptr);
      } else {
        // We do not cache blocks whose size is bigger than the max cache size.
        FreeBlock(ptr, alloc_blocks);
      }
    }
  }

 private:
  explicit TensorAllocator(size_t max_size) : max_size_(max_size) {}

  void* NewBlock(AllocBlocks* alloc_blocks) {
    // We allocate an extra alignment sized area to store the AllocBlocks
    // pointer.
    void* ptr = ::aligned_alloc(
        alloc_blocks->alloc_key.alignment,
        alloc_blocks->alloc_key.alignment + alloc_blocks->alloc_key.num_bytes);
    XLA_CHECK(ptr != nullptr);
    ptr = reinterpret_cast<char*>(ptr) + alloc_blocks->alloc_key.alignment;
    // Store the pointer to AllocBlocks right before the user memory.
    reinterpret_cast<AllocBlocks**>(ptr)[-1] = alloc_blocks;
    size_ += alloc_blocks->alloc_key.num_bytes;
    return ptr;
  }

  void FreeBlock(void* ptr, AllocBlocks* alloc_blocks) {
    size_ -= alloc_blocks->alloc_key.num_bytes;
    std::free(reinterpret_cast<char*>(ptr) - alloc_blocks->alloc_key.alignment);
  }

  void TrimCache(size_t num_bytes) {
    auto it = alloc_list_.rbegin();
    for (; size_ + num_bytes > max_size_ && it != alloc_list_.rend(); ++it) {
      AllocBlocks* alloc_blocks = &*it;
      while (!alloc_blocks->blocks.empty() && size_ + num_bytes > max_size_) {
        FreeBlock(alloc_blocks->blocks.back(), alloc_blocks);
        alloc_blocks->blocks.pop_back();
      }
    }
  }

  size_t max_size_ = 0;
  std::mutex lock_;
  size_t size_ = 0;
  AllocList alloc_list_;
  std::unordered_map<AllocKey, AllocList::iterator, AllocKey::Hash> allocs_;
};

string StripPrefix(const string& value, const string& prefix) {
  return value.find(prefix) == 0 ? value.substr(prefix.size()) : value;
}

tensorflow::DeviceNameUtils::ParsedName ParseFullXrtDevice(
    const string& device) {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  XLA_CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task && parsed_device.has_id &&
      parsed_device.has_type)
      << device;
  return parsed_device;
}

}  // namespace

void XrtComputationClient::XrtData::Assign(const Data& data) {
  const XrtData& xrt_data = dynamic_cast<const XrtData&>(data);
  if (&xrt_data != this) {
    handle_ptr = xrt_data.handle_ptr;
  }
}

XrtComputationClient::XrtComputationClient(
    Options options,
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto)
    : options_(std::move(options)),
      compilation_cache_(sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 64)),
      rng_seed_(0x5a2d296e9) {
  tensorflow::ConfigProto config = CreateConfigProto(options_);
  session_cache_ = absl::make_unique<XrtSessionCache>(
      config, [this](XrtSession* s) { InitSession(s); });
  alloc_session_cache_ = absl::make_unique<XrtSessionCache>(config, nullptr);

  auto default_device_target =
      options_.global_device_map.find(options_.default_device);
  XLA_CHECK(default_device_target != options_.global_device_map.end())
      << options_.default_device;
  for (auto& device : options_.devices) {
    XLA_CHECK(options_.global_device_map.find(device) !=
              options_.global_device_map.end())
        << "Missing device in global map: " << device;
  }
  for (const auto& dev_target : options_.global_device_map) {
    const char* tag =
        options_.devices.count(dev_target.first) > 0 ? "LOCAL" : "REMOTE";
    TF_LOG(INFO) << "XRT device (" << tag << ") " << dev_target.first << " -> "
                 << dev_target.second;
  }
  for (auto& worker_target : options_.workers_map) {
    TF_LOG(INFO) << "Worker " << worker_target.second
                 << " for /job:" << worker_target.first.name
                 << "/replica:0/task:" << worker_target.first.task_no;
  }
  TF_LOG(INFO) << "XRT default device: " << options_.default_device;
  MaybeCreateLocalService(options_);
  InitializeDevices(std::move(topology_proto));
  StartHandleReleaser();
}

ComputationClient::DataPtr XrtComputationClient::CreateDataPlaceholder(
    string device, Shape shape) {
  return std::make_shared<XrtData>(this, std::move(device), std::move(shape));
}

std::vector<ComputationClient::DataPtr> XrtComputationClient::TransferToServer(
    tensorflow::gtl::ArraySlice<const TensorSource> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());

  std::mutex lock;
  XrtSessionCache::SessionMap session_map;
  int64 total_size = 0;
  util::MultiWait mwait(tensors.size());
  std::map<XrtSession*, SessionWork> session_work_map;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto converter = [&, i]() {
      string device = GetEffectiveDevice(tensors[i].device);
      const string& xrt_device = TorchDeviceToXrtDevice(device);
      tensorflow::Tensor tensor(
          TensorAllocator::Get(),
          XlaTypeToDataType(tensors[i].shape.element_type()),
          MakeEquivalentTensorShape(tensors[i].shape));
      auto tdata = tensor.tensor_data();
      tensors[i].populate_fn(tensors[i], const_cast<char*>(tdata.data()),
                             tdata.size());

      {
        std::lock_guard<std::mutex> slock(lock);
        XrtSession* session = GetSessionForXrtDevice(alloc_session_cache_.get(),
                                                     xrt_device, &session_map);
        SessionWork* session_work = &session_work_map[session];
        tensorflow::Scope device_scope =
            session->root()->WithDevice(xrt_device);
        const XrtSession::CachedNode& cached_node =
            GetAllocateNode(session, device_scope, device, tensors[i].shape);
        session_work->feed_inputs.insert({cached_node.holders[0], tensor});
        session_work->outputs_handles.push_back(cached_node.outputs[0]);
        session_work->index_mapping.push_back(i);

        total_size += tdata.size();
      }
    };
    env::ScheduleClosure(mwait.Completer(std::move(converter)));
  }
  mwait.Wait();

  OutboundDataMetric()->AddSample(total_size);

  std::vector<DataPtr> results(tensors.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session_work.first->session()->Run(
        session_work.second.feed_inputs, session_work.second.outputs_handles,
        &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      size_t li = session_work.second.index_mapping[i];
      results[li] = std::make_shared<XrtData>(
          this, GetEffectiveDevice(tensors[li].device), tensors[li].shape,
          outputs[i].scalar<int64>()());
    }
    CreateDataHandlesCounter()->AddValue(outputs.size());
  }
  return results;
}

std::vector<Literal> XrtComputationClient::TransferFromServer(
    tensorflow::gtl::ArraySlice<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());

  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  for (size_t i = 0; i < handles.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*handles[i]);
    XrtSession* session = GetSessionForDevice(session_cache_.get(),
                                              xrt_data.device(), &session_map);
    SessionWork* session_work = &session_work_map[session];
    tensorflow::Scope device_scope =
        session->root()->WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
    const XrtSession::CachedNode& cached_node =
        GetReadNode(session, device_scope, xrt_data.device());
    session_work->feed_inputs.insert(
        {cached_node.holders[0], xrt_data.get_handle()});
    session_work->outputs_handles.push_back(cached_node.outputs[0]);
    session_work->index_mapping.push_back(i);
  }

  int64 total_size = 0;
  std::vector<Literal> results(handles.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session_work.first->session()->Run(
        session_work.second.feed_inputs, session_work.second.outputs_handles,
        &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      size_t li = session_work.second.index_mapping[i];
      LiteralProto response;
      XLA_CHECK(response.ParseFromString(outputs[i].scalar<string>()()));
      results[li] = std::move(Literal::CreateFromProto(response).ValueOrDie());
      total_size += results[li].size_bytes();
    }
  }
  InboundDataMetric()->AddSample(total_size);
  return results;
}

std::vector<ComputationClient::ComputationPtr> XrtComputationClient::Compile(
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());

  std::mutex lock;
  util::MultiWait mwait(instances.size());
  std::vector<ProgramShape> program_shapes(instances.size());
  std::vector<ComputationPtr> results(instances.size());
  std::vector<CompilationCacheKey> cache_keys(instances.size());
  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  for (size_t i = 0; i < instances.size(); ++i) {
    auto builder = [&, this, i]() {
      const CompileInstance& instance = instances[i];
      std::unique_ptr<xrt::XLAComputation> xrt_computation =
          CreateXrtComputation(instance.computation, instance.devices,
                               instance.output_shape);
      CompilationCacheKey cache_key(
          GetResourceDomain(instance.compilation_device),
          xrt_computation->SerializeAsString());
      auto computation_ptr = compilation_cache_.Get(cache_key);
      if (computation_ptr == nullptr) {
        cache_keys[i] = std::move(cache_key);
        program_shapes[i] =
            ProgramShape(xrt_computation->config().program_shape());

        const string& xrt_device =
            TorchDeviceToXrtDevice(instance.compilation_device);
        {
          std::lock_guard<std::mutex> slock(lock);
          XrtSession* session = GetSessionForXrtDevice(
              session_cache_.get(), xrt_device, &session_map);
          SessionWork* session_work = &session_work_map[session];
          tensorflow::Scope device_scope =
              session->root()->WithDevice(xrt_device);
          const XrtSession::CachedNode& cached_node = GetCompileNode(
              session, device_scope, instance.compilation_device);
          session_work->feed_inputs.insert(
              {cached_node.holders[0], cache_keys[i].serialized_computation});
          session_work->outputs_handles.push_back(cached_node.outputs[0]);
          session_work->index_mapping.push_back(i);
        }
      } else {
        results[i] = computation_ptr;
      }
    };
    env::ScheduleClosure(mwait.Completer(std::move(builder)));
  }
  mwait.Wait();
  mwait.Reset(session_work_map.size());

  for (auto& session_and_work : session_work_map) {
    XrtSession* session = session_and_work.first;
    const SessionWork& session_work = session_and_work.second;

    auto session_runner = [&, this, session]() {
      std::vector<tensorflow::Tensor> outputs;
      CheckCompileStatus(
          session->session()->Run(session_work.feed_inputs,
                                  session_work.outputs_handles, &outputs),
          instances, session_work);
      XLA_CHECK_EQ(outputs.size(), session_work.outputs_handles.size());

      size_t output_index = 0;
      for (auto li : session_work.index_mapping) {
        CompileInstance* instance = &instances[li];
        results[li] = std::make_shared<XrtComputation>(
            this, std::move(instance->computation), program_shapes[li],
            std::move(instance->devices),
            outputs[output_index].scalar<int64>()(),
            instance->compilation_device);
        ++output_index;

        compilation_cache_.Add(std::move(cache_keys[li]), results[li]);
        CreateCompileHandlesCounter()->AddValue(1);
      }
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(session_runner)));
  }
  mwait.Wait();
  return results;
}

void XrtComputationClient::CheckCompileStatus(
    const Status& status, const std::vector<CompileInstance>& instances,
    const SessionWork& session_work) {
  if (!status.ok()) {
    std::vector<const XlaComputation*> computations;
    for (auto li : session_work.index_mapping) {
      computations.push_back(&instances[li].computation);
    }
    util::ReportComputationError(status, computations);
  }
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::ExecuteComputation(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const DataPtr> arguments, const string& device,
    const ExecuteComputationOptions& options) {
  metrics::TimedSection timed(ExecuteMetric());

  XrtSessionCache::SessionMap session_map;
  string effective_device = GetEffectiveDevice(device);
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
      &session_map, dynamic_cast<const XrtComputation&>(computation),
      BuildParallelArguments(arguments), options.explode_tuple,
      {effective_device}, &feed_inputs);

  XrtSession* session =
      GetSessionForDevice(session_cache_.get(), effective_device, &session_map);
  std::vector<tensorflow::Tensor> outputs;
  util::CheckComputationStatus(
      session->session()->Run(feed_inputs, {exec_ops.front()}, &outputs),
      {&computation.computation()});
  XLA_CHECK_EQ(outputs.size(), 1);

  return GetComputationResults(outputs[0], computation.program_shape().result(),
                               effective_device);
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::ExecuteReplicated(
    const Computation& computation,
    const std::vector<std::vector<DataPtr>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    const ExecuteReplicatedOptions& options) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
      &session_map, dynamic_cast<const XrtComputation&>(computation), arguments,
      options.explode_tuple, devices, &feed_inputs);
  std::vector<const Computation*> computations(devices.size());
  std::fill(computations.begin(), computations.end(), &computation);

  return RunComputations(session_map, exec_ops, computations, devices,
                         feed_inputs);
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::RunComputations(
    const XrtSessionCache::SessionMap& session_map,
    const std::vector<tensorflow::Output>& exec_ops,
    tensorflow::gtl::ArraySlice<const Computation* const> computations,
    tensorflow::gtl::ArraySlice<const string> devices,
    const tensorflow::ClientSession::FeedType& feed_inputs) {
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
  std::map<XrtSession*, std::vector<size_t>> session_replicas;
  for (size_t i = 0; i < devices.size(); ++i) {
    auto worker_hostport = GetWorkerForDevice(GetEffectiveDevice(devices[i]));
    XrtSession* session = session_map.at(worker_hostport.second).get();
    session_replicas[session].push_back(i);
  }
  XLA_CHECK_EQ(computations.size(), devices.size());

  util::MultiWait mwait(session_replicas.size());
  std::vector<std::vector<DataPtr>> results(devices.size());
  for (auto& sess_replica : session_replicas) {
    XrtSession* session = sess_replica.first;
    const std::vector<size_t>& replicas = sess_replica.second;

    auto session_runner = [&, this, session]() {
      std::vector<tensorflow::Output> exec_nodes;
      std::vector<const XlaComputation*> xla_computations;
      for (auto replica : replicas) {
        exec_nodes.push_back(exec_ops[replica]);
        xla_computations.push_back(&computations[replica]->computation());
      }
      std::vector<tensorflow::Tensor> outputs;
      util::CheckComputationStatus(
          session->session()->Run(feed_inputs, exec_nodes, &outputs),
          xla_computations);
      XLA_CHECK_EQ(outputs.size(), exec_nodes.size());

      for (size_t i = 0; i < outputs.size(); ++i) {
        auto replica = replicas[i];
        results[replica] = GetComputationResults(
            outputs[i], computations[replica]->program_shape().result(),
            GetEffectiveDevice(devices[replica]));
      }
    };
    env::ScheduleIoClosure(mwait.Completer(std::move(session_runner)));
  }
  mwait.Wait();
  return results;
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::ExecuteParallel(
    tensorflow::gtl::ArraySlice<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    const ExecuteParallelOptions& options) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<tensorflow::Output> exec_ops =
      CreateExecuteOps(&session_map, computations, arguments,
                       options.explode_tuple, devices, &feed_inputs);
  return RunComputations(session_map, exec_ops, computations, devices,
                         feed_inputs);
}

std::vector<ComputationClient::DataPtr> XrtComputationClient::ExecuteChained(
    tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
    const string& device) {
  static int64 split_mode = sys_util::GetEnvInt("XRT_SPLIT_CHAINED_EXEC", 0);
  return split_mode ? ExecuteChainedSplit(ops, device)
                    : ExecuteChainedXrt(ops, device);
}

std::vector<ComputationClient::DataPtr> XrtComputationClient::ExecuteChainedXrt(
    tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
    const string& device) {
  metrics::TimedSection timed(ExecuteChainedMetric());

  XrtSessionCache::SessionMap session_map;
  string effective_device = GetEffectiveDevice(device);
  const string& xrt_device = TorchDeviceToXrtDevice(effective_device);
  tensorflow::ClientSession::FeedType feed_inputs;
  XrtSession* session =
      GetSessionForXrtDevice(session_cache_.get(), xrt_device, &session_map);
  tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);

  xrt::XRTChainedExecuteConfig config;
  config.set_core_index_in_replica(0);
  config.set_rng_seed(rng_seed_);

  xrt::XRTChainedExecutePlan plan;
  std::vector<xla::Shape> result_shapes;
  for (size_t i = 0; i < ops.size(); ++i) {
    const ExecuteChainedOp& op = ops[i];
    xrt::XRTChainedExecuteOp* plan_op = plan.add_ops();
    const xla::Shape* op_shape = nullptr;
    if (op.device_data != nullptr) {
      const XrtData& xrt_data = dynamic_cast<const XrtData&>(*op.device_data);
      op_shape = &xrt_data.shape();
      plan_op->set_data_handle(xrt_data.get_handle());
    } else {
      const XrtComputation& xrt_computation =
          dynamic_cast<const XrtComputation&>(*op.computation);
      op_shape = &xrt_computation.program_shape().result();
      plan_op->set_computation_handle(xrt_computation.get_handle());
      for (auto& input : op.inputs) {
        XLA_CHECK_LT(input.op_index, i);

        xrt::XRTChainedExecuteOp::Input* plan_input = plan_op->add_inputs();
        plan_input->set_op_index(input.op_index);
        if (input.output_index) {
          plan_input->set_output_index(*input.output_index + 1);
        }
      }
    }
    for (auto& output : op.outputs) {
      XLA_CHECK(op_shape != nullptr);

      xrt::XRTChainedExecuteOp::Output* plan_output = plan_op->add_outputs();
      plan_output->set_result_index(output.result_index);
      if (output.result_index >= result_shapes.size()) {
        result_shapes.resize(output.result_index + 1);
      }
      if (output.output_index) {
        plan_output->set_output_index(*output.output_index + 1);
        result_shapes[output.result_index] =
            ShapeUtil::GetTupleElementShape(*op_shape, *output.output_index);
      } else {
        result_shapes[output.result_index] = *op_shape;
      }
    }
  }

  const XrtSession::CachedNode& cached_node =
      GetExecuteChainedNode(session, device_scope, effective_device);
  feed_inputs.insert({cached_node.holders[0], plan.SerializeAsString()});
  feed_inputs.insert({cached_node.holders[1], config.SerializeAsString()});

  std::vector<tensorflow::Tensor> outputs;
  util::CheckComputationStatus(
      session->session()->Run(feed_inputs, {cached_node.outputs[0]}, &outputs),
      {});
  XLA_CHECK_EQ(outputs.size(), 1);

  std::vector<DataPtr> results;
  auto handles_vec = outputs[0].vec<int64>();
  for (int64 i = 0; i < handles_vec.size(); ++i) {
    results.push_back(std::make_shared<XrtData>(this, effective_device,
                                                std::move(result_shapes.at(i)),
                                                handles_vec(i)));
  }
  CreateDataHandlesCounter()->AddValue(results.size());
  return results;
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::ExecuteChainedSplit(
    tensorflow::gtl::ArraySlice<const ExecuteChainedOp> ops,
    const string& device) {
  metrics::TimedSection timed(ExecuteChainedMetric());

  std::vector<int64> uses(ops.size(), 0);
  for (auto& op : ops) {
    for (auto& input : op.inputs) {
      uses[input.op_index] += 1;
    }
  }
  XrtSessionCache::SessionMap session_map;
  string effective_device = GetEffectiveDevice(device);
  const string& xrt_device = TorchDeviceToXrtDevice(effective_device);
  XrtSession* session =
      GetSessionForXrtDevice(session_cache_.get(), xrt_device, &session_map);
  tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
  std::vector<std::vector<DataPtr>> ops_outputs(ops.size());
  std::vector<DataPtr> results;
  for (size_t i = 0; i < ops.size(); ++i) {
    const ExecuteChainedOp& op = ops[i];
    if (op.device_data != nullptr) {
      ops_outputs[i].push_back(op.device_data);
    } else {
      tensorflow::ClientSession::FeedType feed_inputs;
      std::vector<DataPtr> arguments;
      arguments.reserve(op.inputs.size());
      for (auto& input : op.inputs) {
        XLA_CHECK_LT(input.op_index, i);
        XLA_CHECK_LT(input.output_index.value_or(0),
                     ops_outputs[input.op_index].size());
        arguments.push_back(
            ops_outputs[input.op_index][input.output_index.value_or(0)]);
      }

      std::vector<tensorflow::Output> exec_ops = CreateExecuteOps(
          &session_map, dynamic_cast<const XrtComputation&>(*op.computation),
          BuildParallelArguments(arguments), /*explode_tuple=*/true,
          {effective_device}, &feed_inputs);

      std::vector<tensorflow::Tensor> outputs;
      util::CheckComputationStatus(
          session->session()->Run(feed_inputs, {exec_ops.front()}, &outputs),
          {&op.computation->computation()});
      XLA_CHECK_EQ(outputs.size(), 1);
      ops_outputs[i] = GetComputationResults(
          outputs[0], op.computation->program_shape().result(),
          effective_device);
    }

    for (auto& output : op.outputs) {
      if (output.result_index >= results.size()) {
        results.resize(output.result_index + 1);
      }
      XLA_CHECK_LT(output.output_index.value_or(0), ops_outputs[i].size());
      results[output.result_index] =
          ops_outputs[i][output.output_index.value_or(0)];
    }
    // Drop references to any intermediate result which is not used anymore.
    for (auto& input : op.inputs) {
      uses[input.op_index] -= 1;
      if (uses[input.op_index] == 0) {
        ops_outputs[input.op_index].clear();
      }
    }
    // We can reset the TF op cache here so that we don't keep allocating new
    // TF op nodes on the session graph.
    session->Reset();
  }
  return results;
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::DeconstructTuple(
    tensorflow::gtl::ArraySlice<const DataPtr> tuples) {
  metrics::TimedSection timed(DeconstructTupleMetric());

  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  std::vector<int64> tuple_elements_count(tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[i]);
    XrtSession* session = GetSessionForDevice(session_cache_.get(),
                                              xrt_data.device(), &session_map);
    SessionWork* session_work = &session_work_map[session];
    session_work->index_mapping.push_back(i);

    tensorflow::Scope device_scope =
        session->root()->WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
    int64 count = ShapeUtil::TupleElementCount(xrt_data.shape());
    tuple_elements_count[i] = count;
    for (int64 j = 0; j < count; ++j) {
      const XrtSession::CachedNode& cached_node =
          GetSubTupleNode(session, device_scope, xrt_data.device());
      session_work->feed_inputs.insert(
          {cached_node.holders[0], xrt_data.get_handle()});
      tensorflow::Tensor index_tensor(tensorflow::DT_INT32,
                                      tensorflow::TensorShape({1}));
      index_tensor.flat<tensorflow::int32>()(0) = j;
      session_work->feed_inputs.insert({cached_node.holders[1], index_tensor});
      session_work->outputs_handles.push_back(cached_node.outputs[0]);
    }
  }

  std::vector<std::vector<DataPtr>> results(tuples.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    XLA_CHECK_OK(session_work.first->session()->Run(
        session_work.second.feed_inputs, session_work.second.outputs_handles,
        &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    size_t output_index = 0;
    for (auto li : session_work.second.index_mapping) {
      const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[li]);
      std::vector<DataPtr> tuple_results;
      for (size_t i = 0; i < tuple_elements_count[li]; ++i, ++output_index) {
        tuple_results.push_back(std::make_shared<XrtData>(
            this, xrt_data.device(),
            ShapeUtil::GetTupleElementShape(xrt_data.shape(), i),
            outputs[output_index].scalar<int64>()()));
      }
      results[li] = std::move(tuple_results);
      CreateDataHandlesCounter()->AddValue(tuple_elements_count[li]);
    }
  }
  return results;
}

XrtSession* XrtComputationClient::GetSessionForTarget(
    XrtSessionCache* cache, const string& target,
    XrtSessionCache::SessionMap* session_map) {
  return cache->GetSession(target, session_map);
}

XrtSession* XrtComputationClient::GetSessionForXrtDevice(
    XrtSessionCache* cache, const string& xrt_device,
    XrtSessionCache::SessionMap* session_map) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  return GetSessionForTarget(cache, worker_hostport.second, session_map);
}

XrtSession* XrtComputationClient::GetSessionForDevice(
    XrtSessionCache* cache, const string& device,
    XrtSessionCache::SessionMap* session_map) {
  return GetSessionForXrtDevice(cache, TorchDeviceToXrtDevice(device),
                                session_map);
}

string XrtComputationClient::GetEffectiveDevice(const string& device) const {
  if (device.empty()) {
    return options_.default_device;
  }
  if (device[0] == ':') {
    // Allow devices with ordinal only specification, to expand from the default
    // device type.
    auto pos = options_.default_device.find(':');
    XLA_CHECK_NE(pos, string::npos) << options_.default_device;
    return options_.default_device.substr(0, pos) + device;
  }
  return device;
}

const string& XrtComputationClient::TorchDeviceToXrtDevice(
    const string& device) const {
  auto device_target =
      options_.global_device_map.find(GetEffectiveDevice(device));
  XLA_CHECK(device_target != options_.global_device_map.end())
      << "Unable to find device: " << device;
  return device_target->second;
}

std::unique_ptr<xrt::XLAComputation> XrtComputationClient::CreateXrtComputation(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<const string> devices,
    const Shape* output_shape) const {
  std::unique_ptr<xrt::XLAComputation> xrt_computation(
      new xrt::XLAComputation());
  auto config = xrt_computation->mutable_config();
  config->set_num_cores_per_replica(1);
  if (devices.size() > 1) {
    auto device_assignment = config->mutable_device_assignment();
    auto computation_device = device_assignment->add_computation_devices();
    for (int64 i = 0; i < devices.size(); ++i) {
      const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
      const auto& core_coords = GetDeviceMeshCoords(xrt_device);
      auto replica_device = computation_device->add_replica_devices();
      for (auto coord : core_coords) {
        replica_device->add_value(coord);
      }
    }
    config->set_num_replicas(devices.size());
  }
  *config->mutable_program_shape() =
      computation.GetProgramShape().ValueOrDie().ToProto();
  if (output_shape != nullptr) {
    *config->mutable_program_shape()->mutable_result() =
        output_shape->ToProto();
  }
  *xrt_computation->mutable_hlo_snapshot() =
      std::move(*computation.Snapshot().ConsumeValueOrDie());
  return xrt_computation;
}

tensorflow::Tensor XrtComputationClient::GetArgumentsInputs(
    tensorflow::gtl::ArraySlice<const DataPtr> arguments,
    const string& device) {
  tensorflow::Tensor inputs_tensor(tensorflow::DT_INT64,
                                   tensorflow::TensorShape({arguments.size()}));
  for (size_t i = 0; i < arguments.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*arguments[i]);
    XLA_CHECK_EQ(device, xrt_data.device());
    inputs_tensor.flat<tensorflow::int64>()(i) = xrt_data.get_handle();
  }
  return inputs_tensor;
}

std::vector<tensorflow::Output> XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map,
    tensorflow::gtl::ArraySlice<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
    tensorflow::gtl::ArraySlice<const string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  std::vector<tensorflow::Output> exec_ops;
  for (size_t i = 0; i < computations.size(); ++i) {
    const XrtComputation* xrt_computation =
        dynamic_cast<const XrtComputation*>(computations[i]);
    auto inputs = GetArgumentsInputs(arguments[i], devices[i]);
    const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
    XrtSession* session =
        GetSessionForXrtDevice(session_cache_.get(), xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert(
        {cached_node.holders[0], xrt_computation->get_handle()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(false);
    exec_config.set_return_exploded_tuple(explode_tuple);
    exec_config.set_rng_seed(rng_seed_);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.push_back(cached_node.outputs[0]);
  }
  return exec_ops;
}

std::vector<tensorflow::Output> XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map, const XrtComputation& computation,
    const std::vector<std::vector<DataPtr>>& arguments, bool explode_tuple,
    tensorflow::gtl::ArraySlice<const string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  std::vector<tensorflow::Output> exec_ops;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto inputs = GetArgumentsInputs(arguments[i], devices[i]);
    const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
    XrtSession* session =
        GetSessionForXrtDevice(session_cache_.get(), xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert({cached_node.holders[0], computation.get_handle()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(false);
    exec_config.set_return_exploded_tuple(explode_tuple);
    exec_config.set_rng_seed(rng_seed_);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.push_back(cached_node.outputs[0]);
  }
  return exec_ops;
}

void XrtComputationClient::ReleaseHandles(
    std::vector<DeviceHandle>* handles,
    const std::function<const XrtSession::CachedNode&(
        XrtSession*, const tensorflow::Scope&, const string&)>& op_generator,
    metrics::Metric* timed_metric, metrics::Counter* destroy_counter) {
  std::vector<DeviceHandle> released_handles;
  {
    std::lock_guard<std::mutex> lock(lock_);
    released_handles.swap(*handles);
  }
  if (!released_handles.empty()) {
    metrics::TimedSection timed(timed_metric);

    XrtSessionCache::SessionMap session_map;
    std::map<XrtSession*, std::vector<DeviceHandle>> session_handles_map;
    for (auto& handle : released_handles) {
      XrtSession* session = GetSessionForDevice(session_cache_.get(),
                                                handle.device, &session_map);
      session_handles_map[session].push_back(handle);
    }
    for (const auto& session_and_handles : session_handles_map) {
      XrtSession* session = session_and_handles.first;
      const std::vector<DeviceHandle>& session_handles =
          session_and_handles.second;
      tensorflow::Tensor handles_tensor(
          tensorflow::DT_INT64,
          tensorflow::TensorShape({session_handles.size()}));
      auto flat_handles_tensor = handles_tensor.flat<tensorflow::int64>();
      for (size_t i = 0; i < session_handles.size(); ++i) {
        flat_handles_tensor(i) = session_handles[i].handle;
      }
      tensorflow::Scope device_scope = session->root()->WithDevice(
          TorchDeviceToXrtDevice(session_handles.front().device));
      const XrtSession::CachedNode& cached_node =
          op_generator(session, device_scope, session_handles.front().device);
      tensorflow::ClientSession::FeedType feed_inputs;
      feed_inputs.insert({cached_node.holders[0], handles_tensor});

      std::vector<tensorflow::Tensor> outputs;
      XLA_CHECK_OK(session->session()->Run(
          feed_inputs, {}, {cached_node.operations[0]}, &outputs));
    }
    destroy_counter->AddValue(released_handles.size());
  }
}

void XrtComputationClient::StartHandleReleaser() {
  int64 num_threads = sys_util::GetEnvInt("XLA_HANDLE_RELEASE_THREADS",
                                          options_.devices.size());
  triggered_task_.reset(
      new util::TriggeredTask([this]() { HandleReleaser(); }, num_threads));
}

void XrtComputationClient::HandleReleaser() {
  auto data_op_generator =
      [this](XrtSession* session, const tensorflow::Scope& scope,
             const string& device) -> const XrtSession::CachedNode& {
    return GetReleaseAllocationHandleNode(session, scope, device);
  };
  ReleaseHandles(&released_data_handles_, data_op_generator,
                 ReleaseDataHandlesTimeMetric(), DestroyDataHandlesCounter());

  auto compile_op_generator =
      [this](XrtSession* session, const tensorflow::Scope& scope,
             const string& device) -> const XrtSession::CachedNode& {
    return GetReleaseCompileHandleNode(session, scope, device);
  };
  ReleaseHandles(&released_compile_handles_, compile_op_generator,
                 ReleaseCompileHandlesTimeMetric(),
                 DestroyCompileHandlesCounter());
}

void XrtComputationClient::ReleaseHandle(int64 handle, const string& device,
                                         std::vector<DeviceHandle>* handles) {
  {
    std::lock_guard<std::mutex> lock(lock_);
    handles->push_back({device, handle});
  }
  triggered_task_->Activate();
}

void XrtComputationClient::ReleaseXrtData(const string& device, int64 handle) {
  ReleaseHandle(handle, device, &released_data_handles_);
  ReleaseDataHandlesCounter()->AddValue(1);
}

void XrtComputationClient::ReleaseXrtComputation(
    const string& compilation_device, int64 handle) {
  ReleaseHandle(handle, compilation_device, &released_compile_handles_);
  ReleaseCompileHandlesCounter()->AddValue(1);
}

std::pair<XrtComputationClient::Worker, string>
XrtComputationClient::GetWorkerForXrtDevice(const string& xrt_device) const {
  tensorflow::DeviceNameUtils::ParsedName parsed_device =
      ParseFullXrtDevice(xrt_device);
  auto worker_hostport =
      options_.workers_map.find(Worker(parsed_device.job, parsed_device.task));
  XLA_CHECK(worker_hostport != options_.workers_map.end()) << xrt_device;
  return std::pair<Worker, string>(worker_hostport->first,
                                   worker_hostport->second);
}

std::pair<XrtComputationClient::Worker, string>
XrtComputationClient::GetWorkerForDevice(const string& device) const {
  return GetWorkerForXrtDevice(TorchDeviceToXrtDevice(device));
}

const std::vector<int>& XrtComputationClient::GetDeviceMeshCoords(
    const string& xrt_device) const {
  auto it = device_mesh_coords_.find(xrt_device);
  if (it == device_mesh_coords_.end()) {
    TF_LOG(FATAL) << "Missing mesh coordinates for device: " << xrt_device;
  }
  return it->second;
}

tensorflow::tpu::TopologyProto XrtComputationClient::InitializeAndFetchTopology(
    const string& job, int task_no, const string& worker_host_port,
    const tensorflow::ConfigProto& config) {
  tensorflow::SessionOptions session_options;
  session_options.env = tensorflow::Env::Default();
  session_options.target = worker_host_port;
  session_options.config = config;

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session(root, session_options);
  string system_device = absl::StrCat("/job:", job, "/replica:0/task:", task_no,
                                      "/device:TPU_SYSTEM:0");
  tensorflow::Scope tpu_system_scope = root.WithDevice(system_device);
  const auto unique_name =
      tpu_system_scope.GetUniqueNameForOp("ConfigureDistributedTPU");
  tensorflow::NodeBuilder builder =
      tensorflow::NodeBuilder(unique_name, "ConfigureDistributedTPU")
          .Attr("embedding_config", "")
          .Attr("tpu_embedding_config", "")
          .Attr("is_global_init", false);
  // TODO: Remove this once the new TF build can be relied upon, on the Cloud
  // TPU side.
  const tensorflow::ClusterDef cluster_def = config.cluster_def();
  if (cluster_def.job_size() > 1 ||
      (cluster_def.job_size() == 1 && cluster_def.job()[0].tasks_size() > 1)) {
    builder.Attr("enable_whole_mesh_compilations", true);
  }

  tpu_system_scope.UpdateBuilder(&builder);

  tensorflow::Node* result;
  root.UpdateStatus(builder.Finalize(tpu_system_scope.graph(), &result));
  XLA_CHECK_OK(tpu_system_scope.status());
  root.UpdateStatus(tpu_system_scope.DoShapeInference(result));

  std::vector<tensorflow::Tensor> outputs;
  XLA_CHECK_OK(root.status());
  XLA_CHECK_OK(session.Run({tensorflow::Output(result, 0)}, &outputs));
  XLA_CHECK_EQ(outputs.size(), 1);

  tensorflow::tpu::TopologyProto topology_proto;
  XLA_CHECK(topology_proto.ParseFromString(outputs[0].scalar<string>()()));
  return topology_proto;
}

void XrtComputationClient::InitializeDevices(
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto) {
  bool is_master = topology_proto == nullptr;
  if (is_master) {
    std::set<Worker> tpu_workers;
    for (const auto& dev_target : options_.global_device_map) {
      tensorflow::DeviceNameUtils::ParsedName parsed_device =
          ParseFullXrtDevice(dev_target.second);
      if (parsed_device.type == "TPU") {
        tpu_workers.emplace(parsed_device.job, parsed_device.task);
      }
    }
    if (!tpu_workers.empty()) {
      const Worker& worker = *tpu_workers.begin();
      auto it = options_.workers_map.find(worker);
      XLA_CHECK(it != options_.workers_map.end());

      TF_LOG(INFO) << "Configuring TPU for master worker " << worker.name << ":"
                   << worker.task_no << " at " << it->second;
      tensorflow::tpu::TopologyProto worker_topology_proto =
          InitializeAndFetchTopology(worker.name, worker.task_no, it->second,
                                     session_cache_->GetConfig());
      if (topology_proto == nullptr) {
        topology_proto = absl::make_unique<tensorflow::tpu::TopologyProto>(
            std::move(worker_topology_proto));
      }
    }
    if (topology_proto != nullptr) {
      TF_LOG(INFO) << "TPU topology: " << topology_proto->DebugString();
    }
  }
  for (const auto& dev_target : options_.global_device_map) {
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseFullXrtDevice(dev_target.second);
    if (parsed_device.type != "TPU") {
      continue;
    }
    XLA_CHECK_LE(parsed_device.task, topology_proto->num_tasks());
    XLA_CHECK_LE(parsed_device.id, topology_proto->num_tpu_devices_per_task());
    // The topology proto 'device_coordinates' is a linear list of
    // [num_tasks][devices_per_task][mesh_shape_size] coordinates, where the
    // mesh coordinates are usually [x, y, c] ('x' and 'y' being the spatial
    // chip coordinated and 'c' the core number).
    int64 base_index = parsed_device.task *
                           topology_proto->num_tpu_devices_per_task() *
                           topology_proto->mesh_shape_size() +
                       parsed_device.id * topology_proto->mesh_shape_size();
    std::vector<int> device_mesh_coords(topology_proto->mesh_shape_size());
    for (int i = 0; i < topology_proto->mesh_shape_size(); ++i) {
      device_mesh_coords[i] =
          topology_proto->device_coordinates(base_index + i);
    }
    device_mesh_coords_.insert(
        {dev_target.second, std::move(device_mesh_coords)});
  }

  // Create the mesh service only if we have more than one worker, or if
  // multi-processing is active.
  string mp_device = GetMultiProcessingDevice();
  if (is_master && topology_proto != nullptr &&
      (options_.workers_map.size() > 1 || !mp_device.empty())) {
    CreateMeshService(*topology_proto);
  }
}

void XrtComputationClient::CreateMeshService(
    const tensorflow::tpu::TopologyProto& topology_proto) {
  struct Device {
    string local_name;
    string global_name;
  };

  service::grpc::Config config;
  config.mutable_proto()->CopyFrom(topology_proto);

  std::map<Worker, std::vector<Device>> workers_devices;
  for (const auto& dev_target : options_.global_device_map) {
    tensorflow::DeviceNameUtils::ParsedName parsed_device =
        ParseFullXrtDevice(dev_target.second);
    string local_name = absl::StrCat(parsed_device.type, ":", parsed_device.id);
    workers_devices[Worker(parsed_device.job, parsed_device.task)].push_back(
        {local_name, dev_target.first});
  }
  for (auto& worker_address : options_.workers_map) {
    service::grpc::Worker* worker = config.add_workers();
    worker->set_name(worker_address.first.name);
    worker->set_task_no(worker_address.first.task_no);
    worker->set_address(worker_address.second);
    for (auto& worker_device : workers_devices[worker_address.first]) {
      service::grpc::Device* device = worker->add_devices();
      device->set_local_name(worker_device.local_name);
      device->set_global_name(worker_device.global_name);
    }
  }

  string mesh_service_address =
      sys_util::GetEnvString("XRT_MESH_SERVICE_ADDRESS", "localhost:53010");
  TF_LOG(INFO) << "Creating mesh service bound to " << mesh_service_address;
  mesh_service_ = absl::make_unique<service::MeshService>(mesh_service_address,
                                                          std::move(config));
}

std::vector<ComputationClient::DataPtr>
XrtComputationClient::GetComputationResults(
    const tensorflow::Tensor& xrt_result, const Shape& result_shape,
    const string& device) {
  std::vector<DataPtr> results;
  if (xrt_result.dims() == 1) {
    auto handles_vec = xrt_result.vec<int64>();
    for (int64 i = 0; i < handles_vec.size(); ++i) {
      results.push_back(std::make_shared<XrtData>(
          this, device, ShapeUtil::GetTupleElementShape(result_shape, i),
          handles_vec(i)));
    }
  } else {
    results.push_back(std::make_shared<XrtData>(this, device, result_shape,
                                                xrt_result.scalar<int64>()()));
  }
  CreateDataHandlesCounter()->AddValue(results.size());
  return results;
}

string XrtComputationClient::GetResourceDomain(const string& device) const {
  return GetWorkerForDevice(device).second;
}

string XrtComputationClient::GetDefaultDevice() const {
  return options_.default_device;
}

size_t XrtComputationClient::GetNumDevices() const {
  return options_.devices.size();
}

std::vector<string> XrtComputationClient::GetLocalDevices() const {
  return std::vector<string>(options_.devices.begin(), options_.devices.end());
}

std::vector<string> XrtComputationClient::GetAllDevices() const {
  std::vector<string> devices;
  for (const auto& dev_target : options_.global_device_map) {
    devices.push_back(dev_target.first);
  }
  return devices;
}

void XrtComputationClient::SetReplicationDevices(std::vector<string> devices) {
  g_replication_devices = std::move(devices);
}

const std::vector<string>& XrtComputationClient::GetReplicationDevices() const {
  return g_replication_devices;
}

void XrtComputationClient::SetRngSeed(size_t seed) { rng_seed_ = seed; }

void XrtComputationClient::InitSession(XrtSession* session) const {
  struct InitNode {
    int count;
    const XrtSession::CachedNode& (XrtComputationClient::*node_ctor)(
        XrtSession*, const tensorflow::Scope&, const string&)const;
  } const init_nodes[] = {
      {16, &XrtComputationClient::GetCompileNode},
      {16, &XrtComputationClient::GetExecuteNode},
      {16, &XrtComputationClient::GetExecuteChainedNode},
      {16, &XrtComputationClient::GetReadNode},
      {16, &XrtComputationClient::GetReleaseAllocationHandleNode},
      {16, &XrtComputationClient::GetReleaseCompileHandleNode},
      {16, &XrtComputationClient::GetSubTupleNode},
  };
  auto devices = GetLocalDevices();
  for (auto& device : devices) {
    // HACK: The XRT ops on the remote GRPC service has only recently been
    // enabled, so until TF 1.14 is out, we cannot add XRT ops on CPU.
    // If there is only one device, even if CPU, this is the local session,
    // which carries the XRT op (as we include them in the BUILD).
    if (device.compare(0, 4, "CPU:") == 0 && devices.size() > 1) {
      continue;
    }
    const string& xrt_device = TorchDeviceToXrtDevice(device);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    for (auto& init : init_nodes) {
      for (int i = 0; i < init.count; ++i) {
        (this->*init.node_ctor)(session, device_scope, device);
      }
    }
  }
  session->Reset();
}

const XrtSession::CachedNode& XrtComputationClient::GetCompileNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtCompile");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtCompile_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTCompile(scope, holders[0]).handle,
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetExecuteNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtExecute");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtExecute_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT64,
             tensorflow::ops::Placeholder::Shape({-1}))});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTExecute(scope, holders[0], holders[1],
                                    {tensorflow::Output(holders[2])}),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetExecuteChainedNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtExecuteChained");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtExecuteChained_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTExecuteChained(scope, holders[0], holders[1]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetReadNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtRead");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtRead_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReadLiteral(scope, holders[0]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetAllocateNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device,
    const Shape& shape) const {
  // Create the proper key for the allocation node. Since the node has shape and
  // layouts attributes, these need to be included within the key.
  std::stringstream ss;
  ss << "XRTAllocateFromTensor(" << shape << ")";
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(ss.str(), device));
  if (cache->Empty()) {
    XLA_COUNTER("XRTAllocateFromTensor_Empty", 1);
    tensorflow::TensorShape tensor_shape(shape.dimensions());
    tensorflow::TensorShape equiv_tensor_shape =
        MakeEquivalentTensorShape(shape);
    std::vector<int> layout(shape.layout().minor_to_major().begin(),
                            shape.layout().minor_to_major().end());
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(
            scope, XlaTypeToDataType(shape.element_type()),
            tensorflow::ops::Placeholder::Shape(equiv_tensor_shape))});
    tensorflow::ops::XRTAllocateFromTensor::Attrs alloc_attrs =
        tensorflow::ops::XRTAllocateFromTensor::Layouts(layout);
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTAllocateFromTensor(scope, {holders[0].output},
                                               {tensor_shape}, alloc_attrs),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode&
XrtComputationClient::GetReleaseAllocationHandleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtReleaseAllocationHandle");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtReleaseAllocationHandle_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReleaseAllocationHandle(scope, holders[0]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetReleaseCompileHandleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtReleaseCompileHandle");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtReleaseCompileHandle_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReleaseCompilationHandle(scope, holders[0]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetSubTupleNode(
    XrtSession* session, const tensorflow::Scope& scope,
    const string& device) const {
  static const string op_name("XrtSubTuple");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    XLA_COUNTER("XrtSubTuple_Empty", 1);
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT32,
             tensorflow::ops::Placeholder::Shape({1}))});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTSubTuple(scope, holders[0], holders[1]),
        std::move(holders)));
  }
  return cache->Get();
}

tensorflow::DataType XrtComputationClient::XlaTypeToDataType(
    PrimitiveType dtype) {
  switch (dtype) {
    case PRED:
      return tensorflow::DT_BOOL;
    case S8:
      return tensorflow::DT_INT8;
    case U8:
      return tensorflow::DT_UINT8;
    case S16:
      return tensorflow::DT_INT16;
    case U16:
      return tensorflow::DT_UINT16;
    case S32:
      return tensorflow::DT_INT32;
    case U32:
      return tensorflow::DT_UINT32;
    case S64:
      return tensorflow::DT_INT64;
    case U64:
      return tensorflow::DT_UINT64;
    case F32:
      return tensorflow::DT_FLOAT;
    case F64:
      return tensorflow::DT_DOUBLE;
    case BF16:
      return tensorflow::DT_BFLOAT16;
    default:
      break;
  }
  XLA_ERROR() << "Unable to convert XLA type " << dtype
              << " to tensorflow DataType";
}

tensorflow::TensorShape XrtComputationClient::MakeEquivalentTensorShape(
    const Shape& shape) {
  Shape eqiv_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);
  return tensorflow::TensorShape(eqiv_shape.dimensions());
}

std::vector<std::vector<ComputationClient::DataPtr>>
XrtComputationClient::BuildParallelArguments(
    tensorflow::gtl::ArraySlice<const DataPtr> arguments) {
  std::vector<std::vector<DataPtr>> para_arguments(1);
  para_arguments[0].insert(para_arguments[0].end(), arguments.begin(),
                           arguments.end());
  return para_arguments;
}

tensorflow::ConfigProto XrtComputationClient::CreateConfigProto(
    const Options& options) {
  static const string* const grpc_proto = new string("grpc://");
  tensorflow::ConfigProto config;
  if (options.workers_map.size() > 1) {
    tensorflow::ClusterDef* cluster_def = config.mutable_cluster_def();
    std::map<std::string, tensorflow::JobDef*> jobs;
    for (auto& worker_target : options.workers_map) {
      auto it = jobs.find(worker_target.first.name);
      if (it == jobs.end()) {
        tensorflow::JobDef* job = cluster_def->add_job();
        job->set_name(worker_target.first.name);
        it = jobs.emplace(worker_target.first.name, job).first;
      }
      tensorflow::JobDef* job = it->second;
      (*job->mutable_tasks())[worker_target.first.task_no] =
          StripPrefix(worker_target.second, *grpc_proto);
    }
  }
  return config;
}

void XrtComputationClient::MaybeCreateLocalService(
    const XrtComputationClient::Options& options) {
  static const string* const grpc_root = new string("grpc://localhost:");
  int task_index = -1;
  string job_name;
  string cluster_spec;
  for (auto& worker_target : options.workers_map) {
    if (worker_target.second.compare(0, grpc_root->size(), *grpc_root) == 0 &&
        worker_target.first.name == "localservice") {
      job_name = worker_target.first.name;
      task_index = worker_target.first.task_no;
      cluster_spec = absl::StrCat(
          worker_target.first.name,
          "|localhost:", worker_target.second.substr(grpc_root->size()));
    }
  }
  if (!cluster_spec.empty()) {
    XrtLocalService* service =
        new XrtLocalService(cluster_spec, job_name, task_index);
    service->Start();
  }
}

string XrtComputationClient::GetMultiProcessingDevice() {
  return sys_util::GetEnvString("XRT_MULTI_PROCESSING_DEVICE", "");
}

}  // namespace xla
