#include "plugins/cpu/test_cpu_plugin.h"

#include <iostream>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"

// Use `test` as the platform name instead of `cpu` so torch_xla treats this
// as an unknown device.
PJRT_Error* test_platform_name(PJRT_Client_PlatformName_Args* args) {
  static const std::string platform_name = "test";
  args->platform_name = platform_name.c_str();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

const PJRT_Api* GetPjrtApi() {
  // HACK: The CPU client is created as a constexpr, so const-casting is
  // undefined behavior. Make a non-const copy of the struct so we can override
  // methods. Don't do this for a real plugin.
  static PJRT_Api pjrt_api = *pjrt::cpu_plugin::GetCpuPjrtApi();
  pjrt_api.PJRT_Client_PlatformName = test_platform_name;

  return &pjrt_api;
}
