#include "torch_xla/csrc/runtime/env_vars.h"

namespace torch_xla {
namespace runtime {
namespace env {

const char* const kEnvNumTpu = "TPU_NUM_DEVICES";
const char* const kEnvNumGpu = "GPU_NUM_DEVICES";
const char* const kEnvNumCpu = "CPU_NUM_DEVICES";
const char* const kEnvTpuvmMode = "TPUVM_MODE";
const char* const kEnvPjRtDevice = "PJRT_DEVICE";
const char* const kEnvPjRtTpuMaxInflightComputations =
    "PJRT_TPU_MAX_INFLIGHT_COMPUTATIONS";
const char* const kEnvPjrtAsyncCpuClient = "PJRT_CPU_ASYNC_CLIENT";
const char* const kEnvPjrtAsyncGpuClient = "PJRT_GPU_ASYNC_CLIENT";
const char* const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";
const char* const kEnvInferredTpuLibraryPath = "PTXLA_TPU_LIBRARY_PATH";
const char* const kEnvXpuLibraryPath = "XPU_LIBRARY_PATH";
const char* const kEnvNeuronLibraryPath = "NEURON_LIBRARY_PATH";
const char* const kEnvPjrtDistServiceAddr = "PJRT_DIST_SERVICE_ADDR";
const char* const kEnvPjRtLocalProcessCount = "PJRT_LOCAL_PROCESS_COUNT";
const char* const kEnvPjRtLocalRank = "PJRT_LOCAL_PROCESS_RANK";
const char* const kEnvPjrtAllocatorCudaAsync = "PJRT_ALLOCATOR_CUDA_ASYNC";
const char* const kEnvPjrtAllocatorPreallocate = "PJRT_ALLOCATOR_PREALLOCATE";
const char* const kEnvPjrtAllocatorFraction = "PJRT_ALLOCATOR_FRACTION";
const char* const kEnvPjrtDynamicPlugins = "PJRT_DYNAMIC_PLUGINS";
const char* const kEnvDistSvcHeartbeatIntervalInSec =
    "DIST_SERVICE_HEARTBEAT_INTERVAL_IN_SEC";
const char* const kEnvDistSvcMaxMissingHeartbeats =
    "DIST_SERVICE_MAX_MISSING_HEARTBEATS";
const char* const kEnvDistSvcShutdownTimeoutInMin =
    "DIST_SERVICE_SHUTDOWN_TIMEOUT_IN_MIN";

}  // namespace env
}  // namespace runtime
}  // namespace torch_xla
