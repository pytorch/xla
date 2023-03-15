#include "third_party/xla_client/env_vars.h"

namespace xla {
namespace env {

const char* const kEnvNumTpu = "TPU_NUM_DEVICES";
const char* const kEnvNumGpu = "GPU_NUM_DEVICES";
const char* const kEnvNumCpu = "CPU_NUM_DEVICES";
const char* const kEnvTpuConfig = "XRT_TPU_CONFIG";
const char* const kEnvDeviceMap = "XRT_DEVICE_MAP";
const char* const kEnvWorkers = "XRT_WORKERS";
const char* const kEnvMeshService = "XRT_MESH_SERVICE_ADDRESS";
const char* const kEnvWorldSize = "XRT_SHARD_WORLD_SIZE";
const char* const kEnvMpDevice = "XRT_MULTI_PROCESSING_DEVICE";
const char* const kEnvShardOrdinal = "RT_SHARD_ORDINAL";
const char* const kEnvTpuvmMode = "TPUVM_MODE";
const char* const kEnvPjRtDevice = "PJRT_DEVICE";
const char* const kEnvPjRtTpuMaxInflightComputations =
    "PJRT_TPU_MAX_INFLIGHT_COMPUTATIONS";
const char* const kEnvPjrtAsyncCpuClient = "PJRT_CPU_ASYNC_CLIENT";
const char* const kEnvPjrtAsyncGpuClient = "PJRT_GPU_ASYNC_CLIENT";
const char* const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";
const char* const kEnvPjrtDistServiceAddr = "PJRT_DIST_SERVICE_ADDR";
const char* const kEnvPjRtLocalRank = "PJRT_LOCAL_PROCESS_RANK";

}  // namespace env
}  // namespace xla
