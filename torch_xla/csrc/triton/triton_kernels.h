#ifndef TORCH_XLA_CSRC_TRITON_H_
#define TORCH_XLA_CSRC_TRITON_H_

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/triton/gpu_kernel_helpers.h"
#include "torch_xla/csrc/triton/gpu_vendor.h"
#include "torch_xla/csrc/triton/triton.pb.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace torch_xla::XLA_GPU_NAMESPACE {

void TritonKernelCall(gpuStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status);
XLA_REGISTER_CUSTOM_CALL_TARGET(TritonKernelCall, "CUDA");

class ModuleImage;

class Kernel {
 public:
  Kernel(std::string kernel_name, uint32_t num_warps, uint32_t shared_mem_bytes,
         std::string ptx, std::string ttir, int compute_capability,
         uint32_t cluster_dim_0, uint32_t cluster_dim_1,
         uint32_t cluster_dim_2);

  absl::Status Launch(gpuStream_t stream, uint32_t grid[3], void** params);

  static Kernel FromProto(const jax_triton::TritonKernel& proto);
  jax_triton::TritonKernel ToProto() const;

  // Returns true if we can launch the kernel without crashing.
  bool CanLaunchOnDevice(gpuDevice_t) const;

 private:
  std::string kernel_name_;
  uint32_t block_dim_x_;
  uint32_t shared_mem_bytes_;
  std::string ptx_;
  std::string ttir_;
  int compute_capability_;
  uint32_t cluster_dims_[3];

  ModuleImage* module_image_ = nullptr;
};

class KernelCall {
 public:
  struct Parameter {
    struct Array {
      size_t bytes_to_zero;
      size_t ptr_divisibility;
    };

    static absl::StatusOr<Parameter> FromProto(
        const jax_triton::TritonKernelCall_Parameter& proto);
    jax_triton::TritonKernelCall_Parameter ToProto() const;

    std::variant<Array, bool, int32_t, uint32_t, int64_t, uint64_t, float,
                 double>
        value;
  };

  KernelCall(Kernel kernel, uint32_t grid_0, uint32_t grid_1, uint32_t grid_2,
             std::vector<Parameter> parameters);

  absl::Status Launch(gpuStream_t stream, void** buffers);

  static absl::StatusOr<KernelCall> FromProto(
      const jax_triton::TritonKernelCall& proto);
  jax_triton::TritonKernelCall ToProto() const;

  // Returns true if we can launch the kernel without crashing.
  bool CanLaunchOnDevice(gpuDevice_t) const;

 private:
  Kernel kernel_;
  uint32_t grid_[3];
  std::vector<Parameter> parameters_;
};

class AutotunedKernelCall {
 public:
  struct Config {
    KernelCall kernel_call;
    std::string description;
  };

  AutotunedKernelCall(
      std::string name, std::vector<Config> configs,
      std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases);

  static absl::StatusOr<KernelCall> Autotune(AutotunedKernelCall kernel_call,
                                             gpuStream_t stream,
                                             void** buffers);

  static absl::StatusOr<AutotunedKernelCall> FromProto(
      const jax_triton::TritonAutotunedKernelCall& proto);
  jax_triton::TritonAutotunedKernelCall ToProto() const;

 private:
  std::string name_;
  std::vector<Config> configs_;
  // (input buffer idx, output buffer idx, size)
  std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases_;
};

}  // namespace torch_xla::XLA_GPU_NAMESPACE

#endif  // TORCH_XLA_CSRC_TRITON_H_