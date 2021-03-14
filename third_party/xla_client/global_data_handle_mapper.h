//
// Created by chriso on 2/27/21.
//
#ifndef CEREBRASPYTORCH_GLOBAL_DATA_HANDLE_MAPPER_HH
#define CEREBRASPYTORCH_GLOBAL_DATA_HANDLE_MAPPER_HH

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/core/platform/default/integral_types.h"

#include <memory>
#include <mutex>

namespace xla {

/**
 * @brief GlobalDataHandleMapper handles data mapping between devices
 */
class GlobalDataHandleMapper {
  static constexpr bool verbose = false;

public:
  typedef int64_t handle_t;

  GlobalDataHandleMapper() = default;

  /**
   * @brief Add data mapping to another device
   */
  void AddMapping(const std::string &device, handle_t handle,
                  const ComputationClient::DataPtr &cloned_data_ptr);

  /**
   * @brief Free device-to-device mapping
   */
  ComputationClient::DataPtr FreeMapping(const std::string &device,
                                         handle_t handle, bool free_both);

  /**
   * @brief Get cloned data mapping
   */
  ComputationClient::DataPtr GetMapping(const std::string &device,
                                        handle_t handle) const;

  /**
   * @brief Add result mapping in case an execution result is on one device and
   *        becomes an argument to another device, it must be pulled and then
   *        pushed
   */
  void AddWeakMapping(const std::string &device, handle_t handle);

  /**
   * @brief Has some sort of mapping, but may be empty if result mapping, for
   */
  bool HasMapping(const std::string &device, handle_t handle) const;

private:
  mutable std::recursive_mutex mtx_;
  using HandleAndDevice = std::pair<int64, std::string>;
  std::map<HandleAndDevice, std::set<HandleAndDevice>> handle_map_;
  std::map<HandleAndDevice, ComputationClient::DataPtr> cloned_data_map_;
};

constexpr const int64 TRACE_HANDLE = -1 /* -1 is no trace */;

} // namespace xla

#endif // CEREBRASPYTORCH_GLOBAL_DATA_HANDLE_MAPPER_HH
