#include "torch_xla/csrc/runtime/env_vars.h"

namespace torch_xla {
namespace runtime {
namespace env {

const char* const kEnvNumTpu = "TPU_NUM_DEVICES";
const char* const kEnvNumGpu = "GPU_NUM_DEVICES";
const char* const kEnvNumCpu = "CPU_NUM_DEVICES";
const char* const kEnvLocalWorker = "XRT_LOCAL_WORKER";
const char* const kEnvTpuConfig = "XRT_TPU_CONFIG";
const char* const kEnvDeviceMap = "XRT_DEVICE_MAP";
const char* const kEnvWorkers = "XRT_WORKERS";
const char* const kEnvMeshService = "XRT_MESH_SERVICE_ADDRESS";
const char* const kEnvWorldSize = "XRT_SHARD_WORLD_SIZE";
const char* const kEnvMpDevice = "XRT_MULTI_PROCESSING_DEVICE";
const char* const kEnvHostOrdinal = "XRT_HOST_ORDINAL";
const char* const kEnvShardOrdinal = "XRT_SHARD_ORDINAL";
const char* const kEnvStartService = "XRT_START_LOCAL_SERVER";
const char* const kEnvTpuvmMode = "TPUVM_MODE";
const char* const kEnvPjRtDevice = "PJRT_DEVICE";
const char* const kEnvPjRtTpuMaxInflightComputations =
    "PJRT_TPU_MAX_INFLIGHT_COMPUTATIONS";
const char* const kEnvPjrtAsyncCpuClient = "PJRT_CPU_ASYNC_CLIENT";
const char* const kEnvPjrtAsyncGpuClient = "PJRT_GPU_ASYNC_CLIENT";
const char* const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";
const char* const kEnvXpuLibraryPath = "XPU_LIBRARY_PATH";
const char* const kEnvPjrtDistServiceAddr = "PJRT_DIST_SERVICE_ADDR";
const char* const kEnvPjRtLocalRank = "PJRT_LOCAL_PROCESS_RANK";

}  // namespace env
}  // namespace runtime
}  // namespace torch_xla
