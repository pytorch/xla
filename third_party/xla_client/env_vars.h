#ifndef XLA_CLIENT_ENV_VARS_H_
#define XLA_CLIENT_ENV_VARS_H_

namespace xla {
namespace env {

extern const char* const kEnvNumTpu;
extern const char* const kEnvNumGpu;
extern const char* const kEnvNumCpu;
extern const char* const kEnvLocalWorker;
extern const char* const kEnvTpuConfig;
extern const char* const kEnvDeviceMap;
extern const char* const kEnvWorkers;
extern const char* const kEnvMeshService;
extern const char* const kEnvWorldSize;
extern const char* const kEnvMpDevice;
extern const char* const kEnvHostOrdinal;
extern const char* const kEnvShardOrdinal;
extern const char* const kEnvStartService;
extern const char* const kEnvTpuvmMode;
extern const char* const kEnvPjRtDevice;
extern const char* const kEnvPjRtTpuMaxInflightComputations;
extern const char* const kEnvPjrtAsyncCpuClient;
extern const char* const kEnvPjrtAsyncGpuClient;
extern const char* const kEnvLibtpuLibraryPath;

}  // namespace env
}  // namespace xla

#endif  // XLA_CLIENT_ENV_VARS_H_
