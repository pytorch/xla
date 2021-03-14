#pragma once

#include "tensorflow/compiler/xla/proxy_client/color_output.h"
#include "tensorflow/compiler/xla/proxy_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor_sentinel.h"

#include "torch_xla/csrc/envfile.h"

#include <sys/syscall.h>

#include <map>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stack>

namespace torch_xla {

/**
 * @brief Local hashing state tracking data
 */
struct ProxyHashingState : public torch_xla::HashingState {
  explicit ProxyHashingState(const xla::hash_t &start_hash)
      : HashingState(start_hash){};
  std::vector<size_t> original_indices_;
  xla::hash_t pre_prune_hash_ = 0;
  std::size_t pass_ = 0;
  bool fabric_run_ = false;
  bool known_executable_ =
      false; // optimization when we know this executable already exists
};

/**
 *  _                         _____             _   _             _
 * | |                       / ____|           | | (_)           | |
 * | |      __ _  ____ _   _| (___   ___  _ __ | |_ _ _ __   ___ | |
 * | |     / _` ||_  /| | | |\___ \ / _ \| '_ \| __| | '_ \ / _ \| |
 * | |____| (_| | / / | |_| |____) |  __/| | | | |_| | | | |  __/| |
 * |______|\__,_|/___| \__, |_____/ \___||_| |_|\__|_|_| |_|\___||_|
 *                      __/ |
 *                     |___/
 */
class LazySentinel : public torch_xla::Sentinel {
public:
  typedef xla::hash_t hash_t;

  /**
   * @brief Create a new hashing state object
   */
  virtual std::shared_ptr<torch_xla::HashingState>
  CreateHashingState(const xla::hash_t &start_hash) const override {
    return std::make_shared<ProxyHashingState>(start_hash);
  };

  /**
   * @brief Notification that a MarkStep is beginning
   *
   *        Note: It is possible to have a stable graph without
   *        the sync occuring inside the step marker.
   */
  void NotifyStepMarkerBegin(const std::string &device_str,
                             const std::vector<std::string> &devices) override;

  /**
   * @brief Notification that a MarkStep is ending
   */
  void NotifyStepMarkerEnd() override;

  /**
   * @brief Notification that a SyncTensorsGraph is occuring.  This means that
   *        a tensor sync is imminent for the given thread, which may or may not
   * be the same tensor set/graph as the previous sync.
   */
  std::vector<xla::ComputationClient::DataPtr> NotifyScheduleSyncTensorsGraph(
      std::vector<torch_xla::XLATensor> *tensors,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      torch_xla::XLATensor::SyncTensorCollection *coll,
      std::shared_ptr<xla::ComputationClient::Computation> &computation)
      override;

  /**
   * @brief Called after hashing the post-order and parameters is complete.
   *        If callee wishes to adjust the graph (i.e. some k=sort of rewrite),
   *        then it shall return true.  Callee should return true a maximum of
   * one time per pass (per unique HashingState object).
   *
   *        - Graph stabilization is based upon pre-rewrite hashes.
   *        - Graph rewrites should be deterministic for a given pre-rewrite
   * hash.
   *        - It is legal to modify some aspects of SyncTensorCollection, such
   * as the output tensor indices (in the case that a rewrite alters the graph
   * outputs)
   *
   * Note: Any second call to this function for the same sync is due to
   * recalculating the hashes from the postorder.  Ideally, this would only
   * occur the *first time* that a new hash is considered as "stable" and the
   * "final" hash cached from that point onwards.
   */
  bool
  OnHashingComplete(torch_xla::HashingState &state,
                    std::vector<torch_xla::XLATensor> *tensors,
                    torch_xla::XLATensor::SyncTensorCollection &coll) override;

  /**
   * @brief Called just before the HLO Module is built.  This gives
   *        the callee the opportunity to "tag" the module (via XlaBuilder)
   *        in the case that it should be handled in a special way (i.e.
   *        performed on a different device).
   */
  bool PreProcessHlo(
      xla::XlaBuilder *builder,
      const torch_xla::XLATensor::SyncTensorCollection &coll) override;

  /**
   *  _                       _    _____              __  _
   * | |                     | |  / ____|            / _|(_)
   * | |      ___   ___  __ _| | | |      ___  _ __ | |_  _  __ _
   * | |     / _ \ / __|/ _` | | | |     / _ \| '_ \|  _|| |/ _` |
   * | |____| (_) | (__| (_| | | | |____| (_) | | | | |  | | (_| |
   * |______|\___/ \___|\__,_|_|  \_____|\___/|_| |_|_|  |_|\__, |
   *                                                         __/ |
   *                                                        |___/
   */
  static void SetOutputs(const std::vector<at::Tensor> &output_tensors,
                         bool append);

  // Maybe should be get last mark step device?
  static bool WasMarkStepOnProxy();

  static void
  SetDeviceProxy(const std::string &device, const std::string &address,
                 std::shared_ptr<xla::ComputationClientFactory> client_factory);

  static bool IsInitialized();

private:
  bool IsAllowedOutput(const torch_xla::XLATensor &tensor,
                       torch_xla::XLATensor::SyncTensorCollection &coll,
                       bool *is_restricting);

  static bool IsTrainingThread(pid_t tid);
  static bool IsQualifyingStep(pid_t tid /*, bool or_higher = false*/);
  static void SetAllDevices(const std::vector<std::string> &all_devices);
  static bool HasProxyDevices();
  bool PruneTensors(std::vector<torch_xla::XLATensor> *tensors,
                    torch_xla::XLATensor::SyncTensorCollection &coll);

  //
  // Data
  //
  static std::vector<std::string> proxy_devices_;
  friend struct MarkStepScope;
};

// LazySentinel can track the behavior of multiple
// threads simultaneously.
inline pid_t gettid() { return syscall(__NR_gettid); }

// Steal some debug output functions
// from the TF codebase
using ColorScope = ::xla::torch_xla::ColorScope;
using EnterLeave = ::xla::torch_xla::EnterLeave;
using Color = ::xla::torch_xla::Color;

} // namespace torch_xla
