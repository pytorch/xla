#include "tensorflow/compiler/xla/xla_client/env_vars.h"

namespace xla {
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

}  // namespace env
}  // namespace xla
