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

}  // namespace env
}  // namespace xla

#endif  // XLA_CLIENT_ENV_VARS_H_
