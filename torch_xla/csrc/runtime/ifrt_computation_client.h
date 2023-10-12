#ifndef XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_

#include <torch/csrc/lazy/backend/backend_data.h>

#include <cstdint>
#include <mutex>
#include <shared_mutex>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/util.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/shape.h"

namespace torch_xla {
namespace runtime {

class IfrtComputationClient : public ComputationClient {
 public:
  IfrtComputationClient();

  DataPtr CreateDataPlaceholder(std::string device, xla::Shape shape) override;

  std::vector<DataPtr> GetDataShards(DataPtr data) override;

  DataPtr GetDataShard(DataPtr data, size_t index) override;

  DataPtr WrapDataShards(const std::vector<DataPtr>& shards, std::string device,
                         xla::Shape shape, xla::OpSharding sharding) override;

  std::optional<xla::OpSharding> GetDataSharding(DataPtr handle) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Use XLA replication to re-assemble the sharded data.
  DataPtr ReplicateShardedData(const DataPtr& handle);

  std::vector<xla::Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  DataPtr TransferShardsToServer(absl::Span<const TensorSource> tensor_shards,
                                 std::string device, xla::Shape shape,
                                 xla::OpSharding sharding) override;

  DataPtr CopyToDevice(DataPtr data, std::string dst) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override;

  size_t GetNumDevices() const override;

  std::string GetDefaultDevice() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  int GetProcessIndex() const override { return client_->process_index(); };

  int GetNumProcesses() const override;

  const absl::flat_hash_map<
      std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>&
  GetDeviceAttributes(const std::string& device) override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void PrepareToExit() override { return; };

  void WaitDeviceOps(const std::vector<std::string>& devices) override;

  std::map<std::string, Metric> GetMetrics() const override;

  // NOT IMPLEMENTED

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

 private:
  std::shared_ptr<xla::ifrt::PjRtClient> client_;
  // global_ordinals_ tracks a map from PjRtDeviceId to the device's
  // dense global ordinal.
  std::unordered_map<int, int> global_ordinals_;
  std::unordered_map<std::string, xla::PjRtDevice* const> string_to_device_;
  std::shared_ptr<std::vector<std::string>> replication_devices_;
  std::unordered_map<std::string, std::unique_ptr<std::shared_mutex>>
      device_locks_;

  xla::PjRtDevice* StringToPjRtDevice(const std::string& device);
  std::shared_lock<std::shared_mutex> lock_device_shared(
      const std::string& device);
  std::unique_lock<std::shared_mutex> lock_device(const std::string& device);

  std::string PjRtDeviceToString(xla::PjRtDevice* const device) const;
  std::vector<std::string> PjRtDevicesToString(
      absl::Span<xla::PjRtDevice* const> devices) const;

  struct IfrtData : public Data {
    IfrtData(std::string device, xla::Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    IfrtData(std::string device, xla::Shape device_shape,
             tsl::RCReference<xla::ifrt::Array> buffer,
             std::optional<xla::OpSharding> sharding = std::nullopt)
        : Data(std::move(device), std::move(device_shape)), buffer(buffer), sharding_(sharding) {}

    IfrtData(std::string device, tsl::RCReference<xla::ifrt::Array> buffer)
        : Data(std::move(device),
               xla::ShapeUtil::MakeShape(
                   xla::ifrt::ToPrimitiveType(buffer->dtype()).value(),
                   buffer->shape().dims())),
          buffer(buffer) {}

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
        ss << "  NumShards: " << buffer->sharding().devices().size() << "\n";
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

  // struct PjRtShardedData : public Data {
  //   PjRtShardedData(std::string device, xla::Shape shape) = delete;

  //   PjRtShardedData(std::string device, xla::Shape shape,
  //                   std::vector<std::shared_ptr<PjRtData>> shards,
  //                   xla::OpSharding sharding)
  //       : Data(std::move(device), std::move(shape)),
  //         shards(shards),
  //         sharding(sharding) {}

  //   Handle GetHandle() override {
  //     // Always returns `Handle` of the first shard.
  //     return shards[0]->GetHandle();
  //   }

  //   void Assign(const torch::lazy::BackendData& data) override {
  //     const PjRtShardedData& pjrt_sharded_data =
  //         dynamic_cast<const PjRtShardedData&>(data);
  //     if (&pjrt_sharded_data != this) {
  //       shards = std::move(pjrt_sharded_data.shards);
  //     }
  //   }

  //   bool HasValue() const override {
  //     if (shards.empty()) {
  //       return false;
  //     }

  //     for (auto& shard : shards) {
  //       if (!shard->HasValue()) {
  //         return false;
  //       }
  //     }
  //     return true;
  //   }

  //   std::string ToString() const override {
  //     std::stringstream ss;
  //     ss << "XLAShardedData: \n";
  //     ss << "  Data Device: " << device() << "\n";
  //     ss << "  Data Shape: " << shape().ToString() << "\n";
  //     ss << "  OpSharding: "
  //        << xla::HloSharding::FromProto(sharding)->ToString() << "\n";
  //     ss << "  NumShards: " << shards.size() << "\n";
  //     return ss.str();
  //   }

  //   bool HasSharding() const override { return true; }

  //   xla::OpSharding GetSharding() const override { return sharding; }

  //   std::vector<std::shared_ptr<PjRtData>> shards;
  //   xla::OpSharding sharding;
  // };

  struct IfrtComputation : public Computation {
    IfrtComputation(xla::XlaComputation computation,
                    std::vector<std::string> devices,
                    std::unique_ptr<xla::ifrt::LoadedExecutable> executable)
        : Computation(std::move(computation), std::move(devices)),
          executable(std::move(executable)) {}

    std::unique_ptr<xla::ifrt::LoadedExecutable> executable;
  };
};

}  // namespace runtime
}  // namespace torch_xla
#endif  // XLA_CLIENT_IFRT_COMPUTATION_CLIENT_H_
