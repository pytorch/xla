#ifndef XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_

#include <torch/csrc/lazy/backend/backend_data.h>

#include <cstdint>
#include <mutex>
#include <shared_mutex>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/operation_manager.h"
#include "torch_xla/csrc/runtime/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"

namespace torch_xla {
namespace runtime {

class PjRtComputationClient : public ComputationClient {
 public:
  PjRtComputationClient();
  ~PjRtComputationClient();

  DataPtr CreateDataPlaceholder(
      std::string device, xla::Shape shape,
      std::optional<xla::OpSharding> sharding = std::nullopt) override;

  static DataPtr CreateData(std::string device, xla::Shape shape,
                            std::shared_ptr<xla::PjRtBuffer> pjrt_buffer);

  std::vector<DataPtr> GetDataShards(DataPtr data) override;

  DataPtr GetDataShard(DataPtr data, size_t index) override;

  DataPtr WrapDataShards(absl::Span<const DataPtr> shards, std::string device,
                         xla::Shape shape, xla::OpSharding sharding) override;

  std::optional<xla::OpSharding> GetDataSharding(DataPtr handle) override;

  std::vector<DataPtr> TransferToDevice(
      absl::Span<const std::shared_ptr<const TensorSource>> tensors) override;

  // Reshard and return data sharded by `sharding` spec. This is a no-op if
  // the input sharding spec is identical to the target `sharding` sharding
  // spec.
  // TODO(yeounoh) replace ReplicateShardedData with this.
  std::vector<DataPtr> ReshardData(
      absl::Span<const DataPtr> handles,
      absl::Span<const xla::OpSharding> shardings) override;

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

  std::string SerializeComputation(const ComputationPtr computation) override;

  ComputationPtr DeserializeComputation(const std::string& serialized) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<DataPtr> ExecuteReplicated(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override;

  size_t GetNumDevices() const override;

  std::string GetDefaultDevice() const override;

  torch_xla::DeviceType GetDeviceType() const override {
    return torch_xla::DeviceType(
        absl::AsciiStrToUpper(client_->platform_name()));
  };

  std::string GetDeviceKind(const std::string& device) override;

  xla::PjRtPlatformId GetPlatformID() const override {
    return client_->platform_id();
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      int local_device_id) const override {
    return client_->LookupAddressableDevice(
        xla::PjRtLocalDeviceId(local_device_id));
  }

  std::intptr_t GetCudaStreamForDevice(int local_device_id) const override {
    absl::StatusOr<xla::PjRtDevice*> pjrt_device =
        client_->LookupAddressableDevice(
            xla::PjRtLocalDeviceId(local_device_id));
    XLA_CHECK(pjrt_device.ok()) << "Failed to get a PjRt device.";
    absl::StatusOr<std::intptr_t> stream =
        pjrt_device.value()->GetStreamForExternalReadyEvents();
    XLA_CHECK(stream.ok()) << "Failed to get a stream.";
    return stream.value();
  }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  torch::lazy::hash_t HashCompilationEnv() override;

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

  MemoryInfo GetMemoryInfo(const std::string& device) override;

  std::string PjRtDeviceToString(xla::PjRtDevice* const device) const override;
  std::vector<std::string> PjRtDevicesToString(
      absl::Span<xla::PjRtDevice* const> devices) const;

  void RegisterCustomCall(const std::string& fn_name, void* function_ptr,
                          const std::string& platform) override;

  void OnReadyCallback(DataPtr data,
                       const std::function<void()>& callback) override;

 private:
  std::unique_ptr<xla::PjRtClient> client_;
  std::unique_ptr<XlaCoordinator> coordinator_;
  // global_ordinals_ tracks a map from PjRtDeviceId to the device's
  // dense global ordinal.
  std::unordered_map<int, int> global_ordinals_;
  std::unordered_map<std::string, xla::PjRtDevice* const> string_to_device_;
  std::shared_ptr<std::vector<std::string>> replication_devices_;
  OperationManager operation_manager_;
  tsl::thread::ThreadPool pool_ = tsl::thread::ThreadPool(
      tsl::Env::Default(), "pjrt", std::thread::hardware_concurrency());
  torch::lazy::hash_t comp_env_hash_;

  xla::PjRtDevice* StringToPjRtDevice(const std::string& device);

  struct PjRtData : public Data {
    PjRtData(std::string device, xla::Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    PjRtData(std::string device, xla::Shape device_shape,
             std::shared_ptr<xla::PjRtBuffer> buffer)
        : Data(std::move(device), std::move(device_shape)), buffer(buffer) {}

    PjRtData(std::string device, std::shared_ptr<xla::PjRtBuffer> buffer)
        : Data(std::move(device),
               xla::Shape(buffer->element_type(), buffer->dimensions(),
                          buffer->is_dynamic_dimension(), {})),
          buffer(buffer) {}

    Handle GetHandle() override {
      XLA_CHECK(HasValue())
          << "buffer with shape " << shape().ToString() << " on device "
          << device() << (buffer == nullptr ? " is null" : " is deleted");
      return reinterpret_cast<std::uintptr_t>(buffer.get());
    };
    void Assign(const torch::lazy::BackendData& data) override;
    bool HasValue() const override {
      return buffer != nullptr && !buffer->IsDeleted();
    };

    bool HasSharding() const override { return false; }

    xla::OpSharding GetSharding() const override {
      XLA_CHECK(false) << "GetSharding should not be called on PjRtData, check "
                          "HasSharding first";
      return xla::OpSharding();
    }

    std::string ToString() const override {
      std::stringstream ss;
      ss << "XLAData: \n";
      ss << "  Data Device: " << device() << "\n";
      ss << "  Data Shape: " << shape().ToString() << "\n";
      ss << "  Data Handle: ";
      if (HasValue()) {
        ss << reinterpret_cast<std::uintptr_t>(buffer.get()) << "\n";
      } else {
        ss << (buffer == nullptr ? "None" : "Deleted") << "\n";
      }
      return ss.str();
    }

    std::shared_ptr<xla::PjRtBuffer> buffer;
  };

  struct PjRtShardedData : public Data {
    PjRtShardedData(std::string device, xla::Shape shape) = delete;

    PjRtShardedData(std::string device, xla::Shape shape,
                    xla::OpSharding sharding)
        : Data(std::move(device), std::move(shape)), sharding(sharding) {}

    PjRtShardedData(std::string device, xla::Shape shape,
                    std::vector<std::shared_ptr<PjRtData>> shards,
                    xla::OpSharding sharding)
        : Data(std::move(device), std::move(shape)),
          shards(shards),
          sharding(sharding) {}

    Handle GetHandle() override {
      // Always returns `Handle` of the first shard.
      return shards[0]->GetHandle();
    }

    void Assign(const torch::lazy::BackendData& data) override {
      const PjRtShardedData& pjrt_sharded_data =
          dynamic_cast<const PjRtShardedData&>(data);
      if (&pjrt_sharded_data != this) {
        shards = std::move(pjrt_sharded_data.shards);
      }
    }

    bool HasValue() const override {
      if (shards.empty()) {
        return false;
      }

      for (auto& shard : shards) {
        if (!shard->HasValue()) {
          return false;
        }
      }
      return true;
    }

    std::string ToString() const override {
      std::stringstream ss;
      ss << "XLAShardedData: \n";
      ss << "  Data Device: " << device() << "\n";
      ss << "  Data Shape: " << shape().ToString() << "\n";
      ss << "  OpSharding: "
         << xla::HloSharding::FromProto(sharding)->ToString() << "\n";
      ss << "  NumShards: " << shards.size() << "\n";
      return ss.str();
    }

    bool HasSharding() const override { return true; }

    xla::OpSharding GetSharding() const override { return sharding; }

    std::vector<std::shared_ptr<PjRtData>> shards;
    xla::OpSharding sharding;
  };

  struct PjRtComputation : public Computation {
    PjRtComputation(xla::XlaComputation computation,
                    std::vector<std::string> devices,
                    std::unique_ptr<xla::PjRtLoadedExecutable> executable)
        : Computation(std::move(computation), std::move(devices)),
          executable(std::move(executable)) {
      output_shardings_ = this->executable->GetOutputShardings();
    }

    const std::string get_memory_info() const override {
      auto memory_stats_status_or = executable->GetCompiledMemoryStats();
      if (memory_stats_status_or.ok()) {
        return memory_stats_status_or.value().DebugString();
      } else {
        return "memory usage is not availiable";
      }
    }

    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
    std::optional<std::vector<xla::OpSharding>> output_shardings_;
  };

  // Use XLA replication to re-assemble the sharded data.
  std::shared_ptr<PjRtData> ReplicateShardedData(const DataPtr& handle);
};

}  // namespace runtime
}  // namespace torch_xla
#endif  // XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
