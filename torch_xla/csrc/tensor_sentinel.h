#pragma once

#include <memory>

#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

/**
 * @brief Base class for hashing analysis.  Sentinel
 *        derivatives may with to override Sentinel::CreateHashingState()
 *        in order to return a more robust object of their own which will
 *        be passed back during tensor hash analysis notifications.
 */
struct HashingState {
  explicit HashingState(const xla::hash_t& start_hash)
      : start_hash_(start_hash){};
  const xla::hash_t start_hash_;
};

class Sentinel {
 public:
  /**
   * @brief Create a new hashing state object
   */
  virtual std::shared_ptr<HashingState> CreateHashingState(
      const xla::hash_t& start_hash) const {
    return std::make_shared<HashingState>(start_hash);
  };

  /**
   * @brief Notification that a MarkStep is beginning
   */
  virtual void NotifyStepMarkerBegin(const std::string& device_str,
                                     const std::vector<std::string>& devices) {}

  /**
   * @brief Notification that a MarkStep is ending
   */
  virtual void NotifyStepMarkerEnd() {}

  /**
   * @brief Notification that a SyncTensorsGraph is occuring.  This means that
   *        a tensor sync is imminent for the given thread, which may or may not
   * be the same tensor set/graph as the previous sync.
   */
  virtual std::vector<xla::ComputationClient::DataPtr>
  NotifyScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      XLATensor::SyncTensorCollection* coll,
      std::shared_ptr<xla::ComputationClient::Computation>& computation) {
    return tensors_data;
  }

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
  virtual bool OnHashingComplete(HashingState& state,
                                 std::vector<XLATensor>* tensors,
                                 XLATensor::SyncTensorCollection& coll) {
    return false;
  }

  /**
   * @brief Called just before the HLO Module is built.  This gives
   *        the callee the opportunity to "tag" the module (via XlaBuilder)
   *        in the case that it should be handled in a special way (i.e.
   *        performed on a different device).
   */
  virtual bool PreProcessHlo(xla::XlaBuilder* builder,
                             const XLATensor::SyncTensorCollection& coll) {
    return false;
  }

  /**
   * @brief Get the current Sentinel
   */
  static std::shared_ptr<Sentinel>& GetSentinel() { return sentinel_; }

  /**
   * @brief Set the Sentinel to use
   */
  static std::shared_ptr<Sentinel> SetSentinel(
      std::shared_ptr<Sentinel> sentinel) {
    auto old_sentinel = sentinel_;
    sentinel_ = std::move(sentinel);
    return old_sentinel;
  }

 private:
  static std::shared_ptr<Sentinel> sentinel_;
};

/**
 * @brief Convenience RAII class for wrapping a MarkStep
 */
struct MarkStepScope {
  MarkStepScope(const std::string& device_str,
                const std::vector<std::string>& devices) {
    Sentinel::GetSentinel()->NotifyStepMarkerBegin(device_str, devices);
  }
  ~MarkStepScope() { Sentinel::GetSentinel()->NotifyStepMarkerEnd(); }
};

}  // namespace torch_xla
