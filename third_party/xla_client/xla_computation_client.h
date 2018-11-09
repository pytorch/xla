#ifndef TENSORFLOW_COMPILER_XLA_RPC_XLA_COMPUTATION_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XLA_COMPUTATION_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace xla {

class XlaComputationClient : public ComputationClient {
  struct XlaData : public Data {
    using Releaser = std::function<void(XlaData*)>;

    XlaData(std::unique_ptr<GlobalData> handle, string device,
            Shape device_shape, Releaser releaser)
        : Data(std::move(device), std::move(device_shape)),
          handle(std::move(handle)),
          releaser(std::move(releaser)) {}

    ~XlaData() override {
      if (releaser) {
        releaser(this);
      }
    }

    std::unique_ptr<GlobalData> Release() {
      CHECK(releaser != nullptr);
      releaser = nullptr;
      return std::move(handle);
    }

    std::unique_ptr<GlobalData> handle;
    Releaser releaser;
  };

 public:
  struct Options {
    Options() : host_name(), port(-1), platform() {}

    string host_name;
    int port;
    string platform;
  };

  XlaComputationClient(Options options);

  std::shared_ptr<Data> TransferParameterToServer(
      const Literal& literal, const string& device) override;

  std::shared_ptr<Data> ExecuteComputation(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments,
      const Shape* output_shape) override;

  std::unique_ptr<Literal> ExecuteComputationAndTransfer(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<Data*> arguments,
      const Shape* output_shape) override;

  std::vector<std::shared_ptr<Data>> ExecuteReplicated(
      const XlaComputation& computation,
      const std::vector<std::vector<Data*>>& arguments,
      const Shape* output_shape) override;

  std::vector<std::shared_ptr<Data>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<const XlaComputation> computations,
      const std::vector<std::vector<Data*>>& arguments,
      tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) override;

  StatusOr<std::vector<std::shared_ptr<Data>>> DeconstructTuple(
      const Data& data) override;

  string GetDefaultDevice() const override;

 private:
  std::vector<GlobalData*> GetArgumentsData(
      tensorflow::gtl::ArraySlice<Data*> arguments, string* device) const;

  // Retrieves the XLA client device handle given the PyTorch device.
  const DeviceHandle& GetDeviceHandle(const string& device) const;

  // Returns the device argument if not empty, or the value returned by the
  // GetDefaultDevice() API.
  string GetEffectiveDevice(const string& device) const;

  // Flushes all the outstanding released handles in one RPC swipe.
  void FlushReleasedHandles();

  // Batches an XLA handle for release.
  void ReleaseXlaData(XlaData* xla_data);

  Options options_;
  Client* client_ = nullptr;
  std::unique_ptr<Client> client_ptr_;
  std::unique_ptr<grpc::XlaService::Stub> xla_service_;
  std::unique_ptr<GRPCStub> stub_;
  std::vector<DeviceHandle> device_handles_;
  std::vector<std::unique_ptr<GlobalData>> released_handles_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XLA_COMPUTATION_CLIENT_H_
