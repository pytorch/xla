#ifndef XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_

#include <cstdint>
#include <mutex>
#include <shared_mutex>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/shape.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/util.h"

namespace xla {

class PjRtComputationClient : public ComputationClient {
 public:
  PjRtComputationClient();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> GetDataShards(DataPtr data) override;

  DataPtr WrapDataShards(const std::vector<DataPtr>& shards, std::string device,
                         xla::Shape shape, xla::OpSharding sharding) override;

  std::optional<xla::OpSharding> GetDataSharding(DataPtr handle) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Use XLA replication to re-assemble the sharded data.
  DataPtr ReplicateShardedData(const DataPtr& handle);

  std::vector<Literal> TransferFromServer(
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

  const absl::flat_hash_map<std::string,
                            xla::ComputationClient::DeviceAttribute>&
  GetDeviceAttributes(const std::string& device) override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void PrepareToExit() override { return; };

  void WaitDeviceOps(const std::vector<std::string>& devices) override;

  // NOT IMPLEMENTED

  void TransferToServer(absl::Span<const TensorSource> tensors,
                        absl::Span<const DataPtr> datas) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<DataPtr> CreateAsyncDatas(
      absl::Span<const TensorSource> tensors) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<xla::util::ExceptionCleanup> LockAsyncDatas(
      absl::Span<const DataPtr> datas) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::string GetResourceDomain(const std::string& device) const override {
    // TODO(wcromar): return a meaningful value
    return "getresourcedomainplaceholder";
  };

  void SetRngSeed(size_t seed) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

  std::map<std::string, Metric> GetMetrics() const override;

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  };

 private:
  std::shared_ptr<PjRtClient> client_;
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

  std::string PjRtDeviceToString(PjRtDevice* const device) const;
  std::vector<std::string> PjRtDevicesToString(
      absl::Span<PjRtDevice* const> devices) const;

  struct PjRtData : public Data {
    PjRtData(std::string device, Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    PjRtData(std::string device, Shape device_shape,
             std::shared_ptr<PjRtBuffer> buffer)
        : Data(std::move(device), std::move(device_shape)), buffer(buffer) {}

    OpaqueHandle GetOpaqueHandle() override {
      XLA_CHECK(HasValue())
          << (buffer == nullptr ? "buffer is null" : "buffer is deleted");
      return reinterpret_cast<std::uintptr_t>(buffer.get());
    };
    void Assign(const Data& data) override;
    bool HasValue() const override {
      return buffer != nullptr && !buffer->IsDeleted();
    };

    std::shared_ptr<PjRtBuffer> buffer;
  };

  struct PjRtShardedData : public Data {
    PjRtShardedData(std::string device, Shape shape) = delete;

    PjRtShardedData(std::string device, Shape shape,
                    std::vector<std::shared_ptr<PjRtData>> shards,
                    xla::OpSharding sharding)
        : Data(std::move(device), std::move(shape)),
          shards(shards),
          sharding(sharding) {}

    OpaqueHandle GetOpaqueHandle() override {
      // Always returns `OpaqueHandle` of the first shard.
      return shards[0]->GetOpaqueHandle();
    }

    void Assign(const Data& data) override {
      const PjRtShardedData& pjrt_sharded_data =
          dynamic_cast<const PjRtShardedData&>(data);
      if (&pjrt_sharded_data != this) {
        shards = std::move(pjrt_sharded_data.shards);
      }
    }

    bool HasValue() const override {
      if (!shards.empty()) {
        for (auto& shard : shards) {
          if (!shard->HasValue()) {
            return false;
          }
        }
      }
      return true;
    }

    xla::OpSharding GetSharding() { return sharding; }

    std::vector<std::shared_ptr<PjRtData>> shards;
    xla::OpSharding sharding;
  };

  struct PjRtComputation : public Computation {
    PjRtComputation(XlaComputation computation, ProgramShape program_shape,
                    std::vector<std::string> devices,
                    std::unique_ptr<xla::PjRtLoadedExecutable> executable)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          executable(std::move(executable)) {}

    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  };
};

}  // namespace xla
#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_H_
