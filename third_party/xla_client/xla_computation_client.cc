#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <csignal>
#include <stdexcept>
#include <string>
#include <strstream>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_client/color_output.h"
#include "tensorflow/compiler/xla/xla_client/global_data_handle_mapper.h"
#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/proxy_name.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_client.h"

namespace xla {

using DataPtr = ComputationClient::DataPtr;
using XrtData = XrtComputationClient::XrtData;
using Computation = ComputationClient::Computation;
using ComputationPtr = ComputationClient::ComputationPtr;
using CompileInstance = ComputationClient::CompileInstance;

namespace {

bool verbose = false;
bool throw_on_compile_fail = true;
bool verbose_transfer = false;
bool verbose_handle_mapping = false;
bool verbose_pull = false;

/**
 * @brief Stub for GRPC versions of xla::ServiceINterface, although
 *        this should still function with a direct pointer to
 *        a local ServiceInterface client.
 */
struct GRPCStubEx : public GRPCStub {
public:
  explicit GRPCStubEx(std::unique_ptr<xla::grpc::XlaService::Stub> stub)
      : GRPCStub(stub.get()) {
    stub_ownership_ = std::move(stub);
  }

private:
  std::unique_ptr<xla::grpc::XlaService::Stub> stub_ownership_;
};

} // end of anonymous namespace

XlaComputationClient::XlaComputationClient(
    std::shared_ptr<xla::ServiceInterface> service)
    : service_(service) {
  StartHandleReleaser();
}

void XlaComputationClient::PrepareToExit() {
  if (triggered_task_ != nullptr) {
    TF_VLOG(1) << "Waiting Proxy handle releaser thread ...";
    size_t run_id = triggered_task_->Activate();
    triggered_task_->WaitForRun(run_id);
    TF_VLOG(1) << "Waiting Proxy handle releaser thread ... done!";
  }
}

DataPtr XlaComputationClient::CreateDataPlaceholder(std::string device,
                                                    Shape shape) {
  return std::make_shared<XrtData>(std::move(device), std::move(shape));
}

/**
 *  _____        _           _____        _
 * |  __ \      | |         |  __ \      | |
 * | |  | | __ _| |_  __ _  | |__) | ___ | | ___   __ _  ___   ___
 * | |  | |/ _` | __|/ _` | |  _  / / _ \| |/ _ \ / _` |/ __| / _ \
 * | |__| | (_| | |_| (_| | | | \ \|  __/| |  __/| (_| |\__ \|  __/
 * |_____/ \__,_|\__|\__,_| |_|  \_\\___||_|\___| \__,_||___/ \___|
 *
 */
void XlaComputationClient::StartHandleReleaser() {
  static const size_t kMinReleaserThreads = 8;
  // TODO: Clean way to get 'Options' here
  int64 num_threads = sys_util::GetEnvInt(
      "XLA_HANDLE_RELEASE_THREADS",
      std::max<size_t>(/*options_.devices.size()*/ 2, kMinReleaserThreads));
  triggered_task_.reset(
      new util::TriggeredTask([this]() { HandleReleaser(); }, num_threads));
}

XlaComputationClient::~XlaComputationClient() {
  if (triggered_task_) {
    triggered_task_->Stop();
    triggered_task_.reset();
  }
}

void XlaComputationClient::HandleReleaser() {
  auto data_op_generator =
      [this](const std::shared_ptr<xla::ServiceInterface> &s,
             const tensorflow::Scope &,
             const std::string &) -> std::shared_ptr<xla::ServiceInterface> {
    return s;
  };
  ReleaseHandles(&released_data_handles_, data_op_generator,
                 nullptr /*ReleaseDataHandlesTimeMetric()*/,
                 nullptr /*DestroyDataHandlesCounter()*/
  );

  auto compile_op_generator =
      [this](const std::shared_ptr<xla::ServiceInterface> &s,
             const tensorflow::Scope &,
             const std::string &) -> std::shared_ptr<xla::ServiceInterface> {
    return s;
  };
  ReleaseHandles(&released_compile_handles_, compile_op_generator,
                 nullptr /*ReleaseCompileHandlesTimeMetric()*/,
                 nullptr /*DestroyCompileHandlesCounter()*/
  );
}

void XlaComputationClient::ReleaseHandles(
    std::vector<DeviceHandle> *handles,
    const std::function<std::shared_ptr<xla::ServiceInterface>(
        const std::shared_ptr<xla::ServiceInterface> &,
        const tensorflow::Scope &, const std::string &)> &op_generator,
    metrics::Metric *timed_metric, metrics::Counter *destroy_counter) {
  std::vector<DeviceHandle> released_handles;
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    released_handles.swap(*handles);
  }
  if (!released_handles.empty()) {
    XrtSessionCache::SessionMap session_map;
    std::map<std::shared_ptr<xla::ServiceInterface>, std::vector<DeviceHandle>>
        session_handles_map;
    for (auto &handle : released_handles) {
      std::shared_ptr<xla::ServiceInterface> session =
          XlaComputationClient::GetXlaClient(
              ProxyName::is_proxy_device_name(handle.device)
                  ? handle.device
                  : ProxyName::proxy_device_name(handle.device),
              true);
      session_handles_map[session].push_back(handle);
    }
    for (const auto &session_and_handles : session_handles_map) {
      const std::shared_ptr<xla::ServiceInterface> &session =
          session_and_handles.first;
      const std::vector<DeviceHandle> &session_handles =
          session_and_handles.second;

      if (verbose || verbose_handle_mapping) {
        torch_xla::ColorScope grn(torch_xla::Color::FG_GREEN);
        std::cout << "Releasing global data handles: ";
        for (const DeviceHandle &dh : session_handles) {
          std::cout << dh.handle << ", ";
        }
        std::cout << std::endl;
      }
      xla::UnregisterRequest request;
      xla::UnregisterResponse response;
      for (const DeviceHandle &dh : session_handles) {
        request.add_data()->set_handle(dh.handle);
      }
      // Unregister may commonly fail during server shutdown or
      // reset.
      session->Unregister(&request, &response);
    }
    if (destroy_counter) {
      destroy_counter->AddValue(released_handles.size());
    }
  }
}

void XlaComputationClient::ReleaseHandleAsync(
    int64 handle, const std::string &device,
    std::vector<DeviceHandle> *handles) {
  {
    std::lock_guard<std::mutex> lock(task_lock_);
    handles->push_back({device, handle});
  }
  triggered_task_->Activate();
}

void XlaComputationClient::ReleaseDataByHandle(const std::string &device,
                                               int64 handle) {
  ReleaseHandleAsync(handle, device, &released_data_handles_);
}

void XlaComputationClient::ReleaseComputation(
    const std::string &compilation_device, int64 handle) {
  ReleaseHandleAsync(handle, compilation_device, &released_compile_handles_);
}

/**
 *   _____ _  _             _
 *  / ____| |(_)           | |
 * | |    | | _  ___  _ __ | |_  ___
 * | |    | || |/ _ \| '_ \| __|/ __|
 * | |____| || |  __/| | | | |_ \__ \
 *  \_____|_||_|\___||_| |_|\__||___/
 *
 *
 */
std::shared_ptr<ServiceInterface>
XlaComputationClient::CreateServiceClient(const std::string &address) {
  if (verbose) {
    std::cout << "Creating XLA client for server at: " << address << std::endl
              << std::flush;
  }
  std::shared_ptr<ServiceInterface> xla_service = std::make_shared<GRPCStubEx>(
      xla::grpc::XlaService::NewStub(::grpc::CreateChannel(
          address, ::grpc::InsecureChannelCredentials())));
  return xla_service;
}

std::shared_ptr<xla::ServiceInterface>
XlaComputationClient::GetXlaClient(const std::string &device, bool create) {
  assert(ProxyName::is_proxy_device_name(device));
  // TODO: Untangle this cross-dependency
  std::lock_guard<std::recursive_mutex> lk(
      ComputationClientManager::computation_client_map_mtx_);
  auto iter =
      ComputationClientManager::computation_client_info_map_.find(device);
  if (iter == ComputationClientManager::computation_client_info_map_.end()) {
    // No address registered for this device
    std::cout << "No proxy configured for device: " << device << std::endl;
    return nullptr;
  }
  if (!iter->second->xla_client_ && create) {
    iter->second->xla_client_ = CreateServiceClient(iter->second->address_);
    if (iter->second->xla_client_) {
      xla::GetDeviceHandlesRequest request;
      xla::GetDeviceHandlesResponse response;
      request.set_device_count(1 /* arbitrarily chosen as 1 */);
      Status status =
          iter->second->xla_client_->GetDeviceHandles(&request, &response);
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }
      iter->second->device_handles_.reserve(response.device_handles_size());

      const int device_ordinal = std::stoi(split(device, ':')[1]);

      for (const ::xla::DeviceHandle &device_handle :
           response.device_handles()) {
        // Add device to our device list
        assert(iter->second->device_handles_.find(device_ordinal) ==
               iter->second->device_handles_.end());
        iter->second->device_handles_.emplace(device_ordinal, device_handle);

        // Reset the device if supported
        xla::ResetDeviceRequest reset_device_request;
        xla::ResetDeviceResponse reset_device_response;
        *reset_device_request.mutable_device_handle() = device_handle;
        status = iter->second->xla_client_->ResetDevice(&reset_device_request,
                                                        &reset_device_response);
        if (!status.ok()) {
          throw std::runtime_error(status.error_message());
        }
      }
    }
  }
  return iter->second->xla_client_;
}

xla::DeviceHandle
XlaComputationClient::GetDeviceHandle(const std::string &device) {
  if (!service_) {
    throw std::runtime_error("Failed to get XLA client for device");
  }
  std::lock_guard<std::recursive_mutex> lk(
      ComputationClientManager::computation_client_map_mtx_);
  auto iter =
      ComputationClientManager::computation_client_info_map_.find(device);
  if (iter == ComputationClientManager::computation_client_info_map_.end()) {
    // No address registered for this device
    throw std::runtime_error("No proxy configured for device");
  }
  std::shared_ptr<XlaClientInfo> info = iter->second;
  const int64 ordinal = GetDeviceOrdinal(device);
  auto found = info->device_handles_.find(ordinal);
  if (found == info->device_handles_.end()) {
    std::stringstream ss;
    ss << "Attempt to get handle of device with unknown ordinal: " << ordinal;
    throw std::runtime_error(ss.str());
  }
  return found->second;
}

/**
 *  _______                      __
 * |__   __|                    / _|
 *    | |_ __  __ _ _ __   ___ | |_  ___  _ __
 *    | | '__|/ _` | '_ \ / __||  _|/ _ \| '__|
 *    | | |  | (_| | | | |\__ \| | |  __/| |
 *    |_|_|   \__,_|_| |_||___/|_|  \___||_|
 *
 *
 */
ComputationClient::DataPtr
XlaComputationClient::TransferLiteralToServer(const std::string &device,
                                              const Literal &literal) {
  xla::TransferToServerRequest request;
  xla::TransferToServerResponse response;

  *request.mutable_literal() = literal.ToProto();

  *request.mutable_device_handle() = GetDeviceHandle(device);

  Status status = service_->TransferToServer(&request, &response);
  if (!status.ok()) {
    throw std::runtime_error(status.error_message());
  }

  if (verbose_transfer || verbose) {
    torch_xla::ColorScope clr(torch_xla::Color::FG_GREEN);
    std::cout << "TransferLiteralToServer() Sent data , received handle: "
              << response.data().handle()
              << ", shape=" << literal.shape().ToString() << std::endl;
  }

  return std::make_shared<XrtData>(this, ProxyName::unproxy_device_name(device),
                                   device, // probably not necessary
                                   literal.shape(), response.data().handle());
}

// Transfers local tensor values to the TPU servers and fetches the handles.
std::vector<DataPtr>
XlaComputationClient::TransferToServer(absl::Span<const TensorSource> tensors) {
  // TODO: Use MultiWait and send multiple in parallel
  std::vector<DataPtr> results;
  results.reserve(tensors.size());
  for (const TensorSource &tensor : tensors) {
    results.emplace_back(
        TransferLiteralToServer(tensor.device, tensor_to_literal(tensor)));
  }
  return results;
}

// Reads the tensor literal values stored at TPU server sites, behind the
// supplied handles.
std::vector<Literal>
XlaComputationClient::TransferFromServer(absl::Span<const DataPtr> handles) {
  if (verbose || verbose_transfer || verbose_pull) {
    for (const DataPtr &d : handles) {
      torch_xla::ColorScope clr(torch_xla::Color::FG_RED);
      std::cout << getpid()
                << " *PROXY* ProxyComputationClient::TransferFromServer() "
                << " handle = " << d->GetOpaqueHandle()
                << ", shape = " << d->shape() << "@" << d->device() << ENDL;
    }
  }

  std::vector<Literal> local_results;
  local_results.reserve(handles.size());
  for (const DataPtr &data_ptr : handles) {
    assert(ProxyName::is_proxy_device_name(data_ptr->device()));
    xla::TransferToClientRequest request;
    xla::TransferToClientResponse response;

    request.mutable_data()->set_handle(data_ptr->GetOpaqueHandle());
    *request.mutable_shape_with_layout() = data_ptr->shape().ToProto();

    Status status = service_->TransferToClient(&request, &response);

    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }

    StatusOr<Literal> result = Literal::CreateFromProto(
        response.literal(), /*prohibit_empty_literal=*/true);

    if (!result.ok()) {
      throw std::runtime_error(result.status().ToString());
    }
    local_results.emplace_back(result.ConsumeValueOrDie());
  }
  return local_results;
}

/**
 *   _____                       _  _
 *  / ____|                     (_)| |
 * | |      ___  _ __ ___  _ __  _ | | ___
 * | |     / _ \| '_ ` _ \| '_ \| || |/ _ \
 * | |____| (_) | | | | | | |_) | || |  __/
 *  \_____|\___/|_| |_| |_| .__/|_||_|\___|
 *                        | |
 *                        |_|
 *
 * Compiles a set of computations.
 */
std::vector<ComputationPtr>
XlaComputationClient::Compile(std::vector<CompileInstance> instances) {
  // WSE (true)
  std::vector<ComputationClient::ComputationPtr> local_results;
  local_results.reserve(instances.size());
  for (CompileInstance &instance : instances) {
    const std::string &compilation_device =
        ProxyName::proxy_device_name(instance.compilation_device);

    if (!service_) {
      throw std::runtime_error("No XLA client for device");
    }

    // Send down to the WSE compiler for the Hlo pass (for now)
    xla::HloModuleProto hlo_module_proto = instance.computation.proto();
    hlo_module_proto = PreProcessHlo(std::move(hlo_module_proto));
    set_frontend_attribute(hlo_module_proto, "framework_platform", "pytorch");

    xla::CompileRequest compile_request;
    xla::CompileResponse compile_response;

    size_t param_num = 0;
    const ProgramShape program_shape =
        instance.computation.GetProgramShape().ValueOrDie();
    for (const Shape &parameter_shape : program_shape.parameters()) {
      torch_xla::ColorScope clr(torch_xla::Color::FG_CYAN);
      if (verbose) {
        std::cout << "Compile: Param " << param_num++
                  << ", shape: " << parameter_shape << std::endl;
      }
      compile_request.add_input_shape_with_layout()->CopyFrom(
          parameter_shape.ToProto());
    }

    *compile_request.mutable_computation() = std::move(hlo_module_proto);
    *compile_request.mutable_execution_options()->add_device_handles() =
        GetDeviceHandle(compilation_device);
    *compile_request.mutable_execution_options()
         ->mutable_shape_with_output_layout() =
        program_shape.result().ToProto();

    const Status status =
        service_->Compile(&compile_request, &compile_response);
    if (status.ok()) {
      if (verbose) {
        std::cout << "computation id: " << compile_response.handle().handle()
                  << " from proto id "
                  << compile_request.mutable_computation()->id() << std::endl;
      }
      // We compiled it ourselves, should insert a
      // ComputationClient::ComputationPtr
      ComputationClient::ComputationPtr computation_ptr =
          std::make_shared<ComputationClient::Computation>(
              XlaComputation(std::move(hlo_module_proto)),
              ProgramShape(instance.computation.proto().host_program_shape()),
              instance.devices, compile_response.handle().handle());
      local_results.push_back(computation_ptr);
    } else {
      std::stringstream ss;
      ss << "The compile failed on the proxy device: " << compilation_device
         << "Reason: " << status.error_message();
      std::cout << "Compile error: " << status.error_message() << std::endl;
      throw std::runtime_error(ss.str());
    }
  }
  return local_results;
}

xla::HloModuleProto
XlaComputationClient::PreProcessHlo(xla::HloModuleProto &&hlo_module_proto) {
  // Base-case does nothing
  return std::move(hlo_module_proto);
}

/**
 *  ______                        _
 * |  ____|                      | |
 * | |__   __  __ ___   ___ _   _| |_  ___
 * |  __|  \ \/ // _ \ / __| | | | __|/ _ \
 * | |____  >  <|  __/| (__| |_| | |_|  __/
 * |______|/_/\_\\___| \___|\__,_|\__|\___|
 *
 * Executes computation with arguments and returns the result.
 * The passed device must match the common device of the arguments Data.
 * If options.explode_tuple is true, the output tuple will be decomposed into
 * its single elements.
 */
std::vector<DataPtr> XlaComputationClient::ExecuteComputation(
    const Computation &computation, absl::Span<const DataPtr> arguments,
    const std::string &_device, const ExecuteComputationOptions &options) {
  // Temporary artifact of code separation: Incoming device name
  // is the proxy device.
  // TODO: These clients shouldn't know anything about dual device names
  const std::string effective_device = _device;
  const std::string device = ProxyName::unproxy_device_name(_device);

  assert(computation.execution_handle() != 0);

  xla::ExecuteRequest request;
  xla::ExecuteResponse response;

  assert(computation.execution_handle());

  request.mutable_handle()->set_handle(computation.execution_handle());

  if (verbose) {
    torch_xla::ColorScope clr(torch_xla::Color::FG_CYAN);
    std::cout << "Proxy Execution handle: " << computation.execution_handle()
              << " " << computation.program_shape().ToString() << std::endl;
  }

  for (auto &dp : arguments) {
    request.add_arguments()->set_handle(dp->GetOpaqueHandle());
  }

  xla::ExecutionOptions eo;
  //*eo.mutable_debug_options() = GetDebugOptionsFromFlags();
  *eo.add_device_handles() = GetDeviceHandle(effective_device);

  Status status = service_->Execute(&request, &response);

  if (!status.ok()) {
    throw std::runtime_error(status.error_message());
  }

  xla::GetShapeRequest gs_request;

  std::vector<ComputationClient::DataPtr> results; // tuple results
  std::vector<xla::GlobalDataHandle> result_handles;

  xla::ShapeProto response_shape;
  gs_request.mutable_data()->set_handle(response.output().handle());
  {
    xla::GetShapeResponse gs_response;
    assert(gs_request.data().handle());
    status = service_->GetShape(&gs_request, &gs_response);
    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }
    response_shape = gs_response.shape();
  }

  if (response_shape.element_type() == xla::PrimitiveType::TUPLE) {
    xla::DeconstructTupleRequest dt_request;
    xla::DeconstructTupleResponse dt_response;

    dt_request.mutable_tuple_handle()->set_handle(response.output().handle());

    status = service_->DeconstructTuple(&dt_request, &dt_response);

    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }

    results.reserve(dt_response.element_handles_size());
    result_handles.reserve(dt_response.element_handles_size());
    for (const ::xla::GlobalDataHandle &element_handle :
         dt_response.element_handles()) {
      result_handles.push_back(element_handle);
    }
    for (const ::xla::GlobalDataHandle &element_handle : result_handles) {
      // TODO(cjolivier01): do in parallel with MultiWait
      ::xla::GetShapeRequest rq;
      ::xla::GetShapeResponse rsp;
      assert(element_handle.handle());
      *rq.mutable_data() = element_handle;
      status = service_->GetShape(&rq, &rsp);
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }
      DataPtr result_data = std::make_shared<XrtData>(
          this, device, ProxyName::proxy_device_name(device),
          Shape(rsp.shape()), element_handle.handle());
      if (verbose) {
        torch_xla::ColorScope clr(torch_xla::Color::FG_BLUE);
        std::cout << "WSE Execution result data: "
                  << result_data->GetOpaqueHandle() << " @ "
                  << result_data->device()
                  << ", shape = " << result_data->shape().ToString()
                  << std::endl;
      }
      results.emplace_back(result_data);
    }
  } else {
    //
    // In the odd case that the result is a single handle rather than a tuple,
    // simply operate on that handle separately
    //
    results.emplace_back(std::make_shared<XrtData>(
        this, device, effective_device, Shape(response_shape),
        response.output().handle()));
  }
  return results;
}

} // namespace xla
