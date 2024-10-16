#ifndef XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_

#include <torch/csrc/lazy/backend/backend_data.h>

#include <cstdint>
#include <mutex>
#include <shared_mutex>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/operation_manager.h"
#include "torch_xla/csrc/runtime/util.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/shape.h"

namespace torch_xla {
namespace runtime {

class IfrtComputationClient : public ComputationClient {
 public:
  IfrtComputationClient();
  ~IfrtComputationClient();

  DataPtr CreateDataPlaceholder(
      std::string device, xla::Shape shape,
      std::optional<xla::OpSharding> sharding = std::nullopt) override;

  std::vector<DataPtr> GetDataShards(DataPtr data) override;

  DataPtr GetDataShard(DataPtr data, size_t index) override;

  DataPtr WrapDataShards(absl::Span<const DataPtr> shards, std::string device,
                         xla::Shape shape, xla::OpSharding sharding) override;

  std::optional<xla::OpSharding> GetDataSharding(DataPtr handle) override;

  std::vector<DataPtr> TransferToDevice(
      absl::Span<const std::shared_ptr<const TensorSource>> tensors) override;

  std::vector<DataPtr> ReshardData(
      absl::Span<const DataPtr> handles,
      absl::Span<const xla::OpSharding> shardings) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<xla::Literal> TransferFromDevice(
      absl::Span<const DataPtr> handles) override;

  std::uintptr_t UnsafeBufferPointer(const DataPtr handle) override;

  std::shared_ptr<xla::PjRtBuffer> GetPjRtBuffer(const DataPtr handle) override;

  DataPtr TransferShardsToDevice(
      absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
      std::string device, xla::Shape shape, xla::OpSharding sharding) override;

  DataPtr CopyToDevice(DataPtr data, std::string dst) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<DataPtr> ExecuteReplicated(
      const Computation& computation, const absl::Span<const DataPtr> arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override;

  size_t GetNumDevices() const override;

  std::string GetDefaultDevice() const override;

  torch_xla::DeviceType GetDeviceType() const override {
    return torch_xla::DeviceType(
        absl::AsciiStrToUpper(client_->platform_name()));
  };

  xla::PjRtPlatformId GetPlatformID() const override {
    return client_->platform_id();
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      int local_device_id) const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::intptr_t GetCudaStreamForDevice(int local_device_id) const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  int GetProcessIndex() const override { return client_->process_index(); };

  int GetNumProcesses() const override;

  const absl::flat_hash_map<
      std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>
  GetDeviceAttributes(const std::string& device) override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void WaitDeviceOps(absl::Span<const std::string> devices = {}) override;

  std::map<std::string, Metric> GetMetrics() const override;

  void InitializeCoordinator(int global_rank, int world_size,
                             std::string master_addr,
                             std::string port) override;

  XlaCoordinator& GetCoordinator() override;

  bool CoordinatorInitialized() const override;

  torch::lazy::hash_t HashCompilationEnv() override { return comp_env_hash_; }

  // NOT IMPLEMENTED

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::string PjRtDeviceToString(xla::PjRtDevice* const device) const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::string SerializeComputation(const ComputationPtr computation) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  ComputationPtr DeserializeComputation(
      const std::string& serialized) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  void RegisterCustomCall(const std::string& fn_name, void* function_ptr,
                          const std::string& platform) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  void OnReadyCallback(DataPtr data,
                       const std::function<void()>& callback) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

 private:
  std::shared_ptr<xla::ifrt::PjRtClient> client_;
  std::unique_ptr<XlaCoordinator> coordinator_;
  // global_ordinals_ tracks a map from PjRtDeviceId to the device's
  // dense global ordinal.
  std::unordered_map<int, int> global_ordinals_;
  std::unordered_map<std::string, xla::ifrt::Device* const> string_to_device_;
  std::shared_ptr<std::vector<std::string>> replication_devices_;
  OperationManager operation_manager_;
  tsl::thread::ThreadPool pool_ = tsl::thread::ThreadPool(
      tsl::Env::Default(), "ifrt", std::thread::hardware_concurrency());
  torch::lazy::hash_t comp_env_hash_;

  xla::ifrt::Device* StringToIfrtDevice(const std::string& device);

  std::string IfrtDeviceToString(xla::ifrt::Device* const device) const;
  std::vector<std::string> IfrtDevicesToString(
      absl::Span<xla::ifrt::Device* const> devices) const;

  struct IfrtData : public Data {
    IfrtData(std::string device, xla::Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    IfrtData(std::string device, xla::Shape device_shape,
             tsl::RCReference<xla::ifrt::Array> buffer,
             std::optional<xla::OpSharding> sharding = std::nullopt)
        : Data(std::move(device), std::move(device_shape)),
          buffer(buffer),
          sharding_(sharding) {}

    IfrtData(std::string device, tsl::RCReference<xla::ifrt::Array> buffer,
             std::optional<xla::OpSharding> sharding = std::nullopt)
        : Data(std::move(device),
               xla::ShapeUtil::MakeShape(
                   xla::ifrt::ToPrimitiveType(buffer->dtype()).value(),
                   buffer->shape().dims())),
          buffer(buffer),
          sharding_(sharding) {}

    Handle GetHandle() override {
      XLA_CHECK(HasValue())
          << "buffer with shape " << shape().ToString() << " on device "
          << device() << (buffer == nullptr ? " is null" : " is deleted");
      return reinterpret_cast<std::uintptr_t>(buffer.get());
    };
    void Assign(const torch::lazy::BackendData& data) override;
    bool HasValue() const override {
      return buffer != nullptr;  // TODO: && !buffer->IsDeleted();
    };

    bool HasSharding() const override { return sharding_.has_value(); }

    xla::OpSharding GetSharding() const override;

    std::string ToString() const override {
      std::stringstream ss;

      if (HasSharding()) {
        ss << "XLAShardedData: \n";
        ss << "  Data Device: " << device() << "\n";
        ss << "  Data Shape: " << shape().ToString() << "\n";
        ss << "  OpSharding: "
           << xla::HloSharding::FromProto(*sharding_)->ToString() << "\n";
        ss << "  NumShards: " << buffer->sharding().devices()->size() << "\n";
      } else {
        ss << "XLAData: \n";
        ss << "  Data Device: " << device() << "\n";
        ss << "  Data Shape: " << shape().ToString() << "\n";
        ss << "  Data Handle: ";
        if (HasValue()) {
          ss << reinterpret_cast<std::uintptr_t>(buffer.get()) << "\n";
        } else {
          ss << "None\n";
        }
      }
      return ss.str();
    }

    std::optional<xla::OpSharding> sharding_;
    tsl::RCReference<xla::ifrt::Array> buffer;
  };

  tsl::RCReference<xla::ifrt::Array> ReplicateShardedData(
      const std::shared_ptr<IfrtData> handle);

  struct IfrtComputation : public Computation {
    IfrtComputation(xla::XlaComputation computation,
                    std::vector<std::string> devices,
                    std::unique_ptr<xla::ifrt::LoadedExecutable> executable)
        : Computation(std::move(computation), std::move(devices)),
          executable(std::move(executable)) {
      output_shardings_ = this->executable->GetOutputShardings();
    }

    std::unique_ptr<xla::ifrt::LoadedExecutable> executable;
    std::optional<std::vector<xla::OpSharding>> output_shardings_;
  };
};

}  // namespace runtime
}  // namespace torch_xla
#endif  // XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_
