#include "tensorflow/compiler/xla/xla_client/pjrt_computation_client.h"

#include <algorithm>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace xla {

namespace {

// TODO(wcromar): relying on the debug string here is probably a bad idea
std::string PjRtDeviceToString(PjRtDevice* const device) {
  std::string str = device->DebugString();
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
  return str;
}

std::vector<std::string> PjRtDevicesToString(
    absl::Span<PjRtDevice* const> devices) {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (size_t i = 0; i < devices.size(); ++i) {
    strs.push_back(PjRtDeviceToString(devices[i]));
  }

  return strs;
}

}  // namespace

PjRtComputationClient::PjRtComputationClient(Options options) {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");
  if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    client = xla::GetCpuClient(/*asynchronous=*/false).ValueOrDie();
  } else {
    XLA_ERROR() << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                   device_type);
  }

  XLA_CHECK(client.get() != nullptr);
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

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    xla::Shape shape = tensors[i].shape;
    xla::Literal literal(shape);
    tensors[i].populate_fn(tensors[i], literal.untyped_data(),
                           literal.size_bytes());

    std::shared_ptr<xla::PjRtBuffer> buffer =
        client->BufferFromHostLiteral(literal, client->addressable_devices()[0])
            .ValueOrDie();
    buffer->BlockHostUntilReady();
    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensors[i].device, tensors[i].shape, buffer);
    datas.push_back(data);
  }

  return datas;
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());

  for (size_t i = 0; i < handles.size(); ++i) {
    const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(*handles[i]);

    std::shared_ptr<xla::Literal> literal =
        pjrt_data.buffer->ToLiteral().ValueOrDie();
    literals.push_back(std::move(*literal));
  }

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<ComputationClient::ComputationPtr> computations;

  for (size_t i = 0; i < instances.size(); ++i) {
    xla::ProgramShape program_shape =
        instances[i].computation.GetProgramShape().ValueOrDie();
    xla::CompileOptions compile_options;
    std::shared_ptr<PjRtComputation> pjrt_computation =
        std::make_shared<PjRtComputation>(
            client.get(), std::move(instances[i].computation), program_shape,
            instances[i].devices, compile_options);

    computations.push_back(pjrt_computation);
  }

  return computations;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  TF_VLOG(1) << "Executing PjRt computation on " << device;
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);

  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(arguments.size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(arguments[i].get());
    buffers.push_back(pjrt_data->buffer.get());
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results =
      pjrt_computation.executable->Execute({buffers}, execute_options)
          .ValueOrDie();

  std::vector<DataPtr> datas;
  datas.reserve(results[0].size());
  for (size_t i = 0; i < results[0].size(); ++i) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(results[0][i]);

    std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
        device, buffer->logical_on_device_shape().ValueOrDie(),
        std::move(buffer));

    datas.push_back(data);
  }

  TF_VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

size_t PjRtComputationClient::GetNumDevices() const {
  return client->addressable_device_count();
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return PjRtDeviceToString(client->addressable_devices()[0]);
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  return PjRtDevicesToString(client->addressable_devices());
}

std::vector<std::string> PjRtComputationClient::GetAllDevices() const {
  return PjRtDevicesToString(client->devices());
}

std::shared_ptr<std::vector<std::string>>
PjRtComputationClient::GetReplicationDevices() {
  return std::make_shared<std::vector<std::string>>(
      PjRtDevicesToString(client->addressable_devices()));
}

}  // namespace xla
