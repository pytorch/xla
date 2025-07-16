// Names of environment variables.

#ifndef XLA_CLIENT_ENV_VARS_H_
#define XLA_CLIENT_ENV_VARS_H_

namespace torch_xla {
namespace runtime {
namespace env {

inline constexpr char kEnvLocalWorker[] = "LOCAL_WORKER";
inline constexpr char kEnvTpuConfig[] = "TPU_CONFIG";
inline constexpr char kEnvNumTpu[] = "TPU_NUM_DEVICES";
inline constexpr char kEnvNumGpu[] = "GPU_NUM_DEVICES";
inline constexpr char kEnvNumCpu[] = "CPU_NUM_DEVICES";
inline constexpr char kEnvTpuvmMode[] = "TPUVM_MODE";
inline constexpr char kEnvPjRtDevice[] = "PJRT_DEVICE";
inline constexpr char kEnvPjRtTpuMaxInflightComputations[] =
    "PJRT_TPU_MAX_INFLIGHT_COMPUTATIONS";
inline constexpr char kEnvPjrtAsyncCpuClient[] = "PJRT_CPU_ASYNC_CLIENT";
inline constexpr char kEnvPjrtAsyncGpuClient[] = "PJRT_GPU_ASYNC_CLIENT";
inline constexpr char kEnvTpuLibraryPath[] = "TPU_LIBRARY_PATH";
inline constexpr char kEnvInferredTpuLibraryPath[] = "PTXLA_TPU_LIBRARY_PATH";
inline constexpr char kEnvXpuLibraryPath[] = "XPU_LIBRARY_PATH";
inline constexpr char kEnvNeuronLibraryPath[] = "NEURON_LIBRARY_PATH";
inline constexpr char kEnvPjrtDistServiceAddr[] = "PJRT_DIST_SERVICE_ADDR";
inline constexpr char kEnvPjRtLocalProcessCount[] = "PJRT_LOCAL_PROCESS_COUNT";
inline constexpr char kEnvPjRtLocalRank[] = "PJRT_LOCAL_PROCESS_RANK";
inline constexpr char kEnvPjrtAllocatorCudaAsync[] =
    "PJRT_ALLOCATOR_CUDA_ASYNC";
inline constexpr char kEnvPjrtAllocatorPreallocate[] =
    "PJRT_ALLOCATOR_PREALLOCATE";
inline constexpr char kEnvPjrtAllocatorFraction[] = "PJRT_ALLOCATOR_FRACTION";
inline constexpr char kEnvPjrtDynamicPlugins[] = "PJRT_DYNAMIC_PLUGINS";
inline constexpr char kEnvDistSvcHeartbeatIntervalInSec[] =
    "DIST_SERVICE_HEARTBEAT_INTERVAL_IN_SEC";
inline constexpr char kEnvDistSvcMaxMissingHeartbeats[] =
    "DIST_SERVICE_MAX_MISSING_HEARTBEATS";
inline constexpr char kEnvDistSvcShutdownTimeoutInMin[] =
    "DIST_SERVICE_SHUTDOWN_TIMEOUT_IN_MIN";

}  // namespace env
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_ENV_VARS_H_
