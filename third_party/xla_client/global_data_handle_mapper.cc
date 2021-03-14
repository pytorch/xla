#include "tensorflow/compiler/xla/xla_client/global_data_handle_mapper.h"
#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_name.h"

namespace xla {

namespace {
bool verbose_handle_mapping = false;
}  // namespace

/**
 * @brief Add data mapping to another device
 */
void GlobalDataHandleMapper::AddMapping(
    const std::string &device, handle_t handle,
    const ComputationClient::DataPtr &cloned_data_ptr) {
  assert(!device.empty() && handle);
  assert(!cloned_data_ptr || device != cloned_data_ptr->device());
  assert(!ProxyName::is_proxy_device_name(device));
  if (cloned_data_ptr) {
    assert(ProxyName::is_proxy_device_name(cloned_data_ptr->device()));
  }
  const HandleAndDevice src{handle, device};
  if (cloned_data_ptr) {
    const HandleAndDevice dest{cloned_data_ptr->GetOpaqueHandle(),
                               cloned_data_ptr->device()};
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    handle_map_[src].insert(dest);
    handle_map_[dest].insert(src);
    cloned_data_map_[src] = cloned_data_ptr;
    cloned_data_map_[dest] = cloned_data_ptr;
    if (verbose || verbose_handle_mapping) {
      std::cout << "Added mapping: " << handle << " @ " << device << " -> "
                << cloned_data_ptr->GetOpaqueHandle() << " @ "
                << cloned_data_ptr->device() << std::endl
                << std::flush;
    }
  } else {
    // Assure there's an entry, although it may be empty
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    handle_map_[src];
  }
}

/**
 * @brief Free device-to-device mapping
 */
ComputationClient::DataPtr GlobalDataHandleMapper::FreeMapping(
    const std::string &device, handle_t handle, bool free_both) {
  assert(!device.empty() && handle);
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  const HandleAndDevice hd{handle, device};
  auto iter = handle_map_.find(hd);
  if (iter != handle_map_.end()) {
    std::set<HandleAndDevice> &mapped_set = iter->second;
    for (auto set_iter : mapped_set) {
      const HandleAndDevice mapped = set_iter;
      handle_map_[mapped].erase(hd);
    }
    handle_map_.erase(iter);
    auto cloned_iter = cloned_data_map_.find(hd);
    if (cloned_iter != cloned_data_map_.end()) {
      ComputationClient::DataPtr p = cloned_iter->second;
      if (p->GetOpaqueHandle() == handle && p->device() == device) {
        if (verbose || verbose_handle_mapping) {
          std::cout << "Freeing via LOCAL mapped: " << p->GetOpaqueHandle()
                    << " @ " << p->device() << std::endl
                    << std::flush;
        }
      } else {
        if (verbose || verbose_handle_mapping) {
          std::cout << "Freeing via MAPPED  mapped: " << p->GetOpaqueHandle()
                    << " @ " << p->device() << std::endl
                    << std::flush;
        }
      }

      if (free_both) {
        ComputationClient::DataPtr cloned_data_ptr = cloned_iter->second;
        assert(cloned_data_ptr);
        const int64_t cloned_handle = cloned_data_ptr->GetOpaqueHandle();
        if (cloned_handle == TRACE_HANDLE) {
          std::cout << "Freeing cloned from FreeHandle: handle = "
                    << cloned_handle << std::endl;
        }
        FreeMapping(cloned_data_ptr->device(), cloned_handle, false);
      }

      cloned_data_map_.erase(cloned_iter);
      return p;
    }
  }
  return nullptr;
}

/**
 * @brief Get cloned data mapping
 */
ComputationClient::DataPtr GlobalDataHandleMapper::GetMapping(
    const std::string &device, handle_t handle) const {
  assert(!device.empty() && handle);
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  const HandleAndDevice hd{handle, device};
  auto iter = cloned_data_map_.find(hd);
  if (iter == cloned_data_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

/**
 * @brief Add result mapping in case an execution result is on one device and
 *        becomes an argument to another device, it must be pulled and then
 *        pushed
 */
void GlobalDataHandleMapper::AddWeakMapping(const std::string &device,
                                            handle_t handle) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  AddMapping(device, handle, nullptr);
}

/**
 * @brief Has some sort of mapping, but may be empty if result mapping, for
 * instance
 */
bool GlobalDataHandleMapper::HasMapping(const std::string &device,
                                        handle_t handle) const {
  // assert(!ProxyName::is_proxy_device_name(device));
  const HandleAndDevice src{handle, device};
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  return handle_map_.find(src) != handle_map_.end();
}

}  // namespace xla
