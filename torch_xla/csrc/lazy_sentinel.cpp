#include "torch_xla/csrc/lazy_sentinel.h"
#include "torch_xla/csrc/tensor_ex.h"

#include <Python.h>

#include <mutex>
#include <stack>
#include <string>

#include "tensorflow/compiler/xla/proxy_client/color_output.h"
#include "tensorflow/compiler/xla/proxy_client/computation_client_manager.h"
#include "tensorflow/compiler/xla/proxy_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/proxy_client/proxy_name.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

using namespace torch_xla;

#if 1 || !defined(NDEBUG) /* We always wantr our asserts to fire */
#define __ASSERT_FUNCTION __extension__ __PRETTY_FUNCTION__

void _my_assert_handler() { raise(SIGTRAP); }

#undef assert
#define assert(expr)                                                           \
  (static_cast<bool>(expr) \
       ? void(0)           \
       : _my_assert_handler() /*__assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION)*/)
#endif

/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {

namespace {
const bool verbose = VERBOSE_FILE(false);
const bool verbose_tensor_sync = verbose;
const bool verbose_output_control = verbose || false;
const bool verbose_mp = false;
const bool verbose_hash = false;
const bool verbose_remove_tensors = false;
const bool verbose_non_fabric = false;
const bool verbose_mark_step = false;
const bool disable_proxy =
    xla::sys_util::GetEnvBool("WSE_DISABLE_PROXY", false);
const bool prune_tensors_if_outputs_set = true;

constexpr std::size_t DEFAULT_CLEAN_STEPS_UNTIL_PROXY = 1;

} // namespace

std::string mp() {
  std::stringstream ss;
  if (verbose_mp) {
    ss << "[pid=" << getpid() << "] ";
  }
  return ss.str();
}

namespace {

constexpr std::size_t INVALID_COUNT = std::numeric_limits<std::size_t>::max();

using Lock = std::lock_guard<std::recursive_mutex>;

struct ThreadCompileInfo {
  ThreadCompileInfo() : hash_(0U) {}
  std::atomic<std::size_t> sync_count_since_hash_change_{INVALID_COUNT};
  std::atomic<std::size_t> mark_step_count_since_last_reset_{INVALID_COUNT};
  std::unordered_set<size_t> output_ids_;

  void set_hash(LazySentinel::hash_t hash) {
    if (hash != hash_.load()) {
      hash_ = std::move(hash);
    }
  }
  LazySentinel::hash_t hash() const { return hash_; }

private:
  std::atomic<LazySentinel::hash_t> hash_;
};

typedef __int128 Int128;

inline Int128 H128(const xla::hash_t &h) {
  return h.operator unsigned __int128();
}

/**
 * @brief Class to keep track of known-good executables
 */
class ExecutableCache {
  void add_executable(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    assert(executables_.find(hh) == executables_.end());
    auto exec = executables_.insert(hh);
  }

public:
  bool get_executable_by_adjusted_hash(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    auto found = adjusted_hash_map_.find(hh);
    if (found != adjusted_hash_map_.end()) {
      return true;
    }
    return false;
  }
  bool has_executable(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    return executables_.count(hh) != 0;
  }
  bool has_executable_by_adjusted_hash(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    return adjusted_hash_map_.count(hh) != 0;
  }
  bool is_active_executable(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    if (has_executable(hh)) {
      return true;
    }
    return false;
  }
  void activate_hash(const LazySentinel::hash_t &hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    auto found = executables_.find(hh);
    if (found == executables_.end()) {
      add_executable(hh);
    }
  }
  void set_adjusted_hash(const xla::hash_t &h1, const xla::hash_t &h2) {
    Lock lk(mtx_);
    assert(h1 != h2);
    const Int128 hh1 = H128(h1);
    const Int128 hh2 = H128(h2);
    auto found = executables_.find(hh1);
    if (found != executables_.end()) {
      // Should only set this once
      auto found_adjusted = adjusted_hash_map_.find(hh1);
      if (found_adjusted != adjusted_hash_map_.end()) {
        assert(found_adjusted->second == hh1);
      } else {
        adjusted_hash_map_[hh2] = hh1;
      }
    } else {
      assert(false); // does this ever happen?
    }
  }

private:
  mutable std::recursive_mutex mtx_;
  std::set<Int128> executables_; // needs to be locked?
  std::map<Int128, Int128> adjusted_hash_map_;
};

std::mutex compile_info_map_mtx_;
std::map<pid_t, std::shared_ptr<ThreadCompileInfo>> compile_info_map;

/**
 * Get the "ThreadCompileInfo" object for a given thread.
 * "ThreadCompileInfo" tracks the sync and step behavior on a per-thread
 * basis.
 */
std::shared_ptr<ThreadCompileInfo> GetCompileInfo(pid_t tid) {
  std::lock_guard<std::mutex> lk(compile_info_map_mtx_);
  std::shared_ptr<ThreadCompileInfo> sp = compile_info_map[tid];
  if (!sp) {
    sp = compile_info_map[tid] = std::make_shared<ThreadCompileInfo>();
  }
  return std::move(sp);
}

std::shared_ptr<ExecutableCache> ex_cache = std::make_shared<ExecutableCache>();

int get_number_of_required_runs_since_reset() {
  if (disable_proxy) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::cerr << "**** WARNING **** PROXY IS DISABLED" << std::endl;
    }
    return std::numeric_limits<int>::max();
  }
  static bool trusted_model =
      xla::sys_util::GetEnvBool("XLA_TRUSTED_MODEL", false);
  if (trusted_model) {
    return 0;
  }
  static int rtc = xla::sys_util::GetEnvInt("XLA_CLEAN_STEPS_UNTIL_PROXY",
                                            DEFAULT_CLEAN_STEPS_UNTIL_PROXY);
  return rtc;
}

std::mutex init_devices_mutex;

bool thread_local is_in_mark_step = false;
bool thread_local is_clean_step = false;
bool thread_local mark_step_was_on_proxy = false;
bool thread_local prev_step_was_on_proxy = false;

} // namespace

std::vector<std::string> LazySentinel::proxy_devices_;

void LazySentinel::SetAllDevices(const std::vector<std::string> &all_devices) {
  proxy_devices_.clear();
  proxy_devices_.reserve(all_devices.size());
  for (const std::string &device_str : all_devices) {
    const Device device(device_str);
    if (device.hw_type == DeviceType::WSE) {
      proxy_devices_.push_back(device_str);
    }
  }
}

bool LazySentinel::PreProcessHlo(xla::XlaBuilder *builder,
                                 const XLATensor::SyncTensorCollection &coll) {
  if (HasProxyDevices() && IsTrainingThread(coll.requesting_tid)) {
    if (verbose) {
      std::cout << "PreProcessHlo(): " << coll.hash << std::endl;
    }
    bool has_adjusted_exe =
        ex_cache->get_executable_by_adjusted_hash(coll.hash);
    if (has_adjusted_exe) {
      if (true /*exe->is_active()*/) {
        // Mark this for proxy
        xla::FrontendAttributes frontend_attributes;
        frontend_attributes.CopyFrom(builder->frontend_attributes());
        (*frontend_attributes.mutable_map())["PROXY_DEVICE"] =
            coll.device.ToString();
        builder->SetFrontendAttributes(frontend_attributes);

        // Sanity check that if we're pruning outputs,
        // the program shape has the same number of outputs as is expected
#ifndef NDEBUG
        std::shared_ptr<ThreadCompileInfo> compile_info =
            GetCompileInfo(coll.requesting_tid);
        const std::size_t output_ids_size = compile_info->output_ids_.empty();
        if (output_ids_size) {
          const xla::Shape &result_shape =
              builder->GetProgramShape().ValueOrDie().result();
          std::size_t output_count;
          if (result_shape.element_type() == xla::PrimitiveType::TUPLE) {
            output_count = result_shape.tuple_shapes_size();
          } else {
            output_count = 1;
          }
          if (output_count != output_ids_size) {
            XLA_ERROR()
                << "We expected the pruned fabric program shape to have "
                << output_ids_size << " outputs, but it actually had "
                << output_count
                << " outputs.  This is probably a bug and should "
                << " be reported to the developers.";
          }
        }
#endif
        return true;
      } else {
        assert(false); // just checking, will it ever not be?
      }
    }
  }
  return false;
}

void LazySentinel::SetDeviceProxy(
    const std::string &device, const std::string &address,
    std::shared_ptr<xla::ComputationClientFactory> client_factory) {
  // Do not create the computation client
  xla::ComputationClientManager::SetDeviceFactory(device, address,
                                                  std::move(client_factory));
}

bool LazySentinel::HasProxyDevices() {
  static bool got_devices = false;
  if (!got_devices) {
    std::lock_guard<std::mutex> lk(init_devices_mutex);
    if (!got_devices) {
      SetAllDevices(xla::XrtComputationClient::Get()->GetAllDevices());
      got_devices = true;
    }
  }
  return !proxy_devices_.empty();
}

bool LazySentinel::PruneTensors(std::vector<XLATensor> *tensors,
                                XLATensor::SyncTensorCollection &coll) {
  if (!tensors || tensors->empty()) {
    return false;
  }
  std::vector<size_t> adjusted_indices;
  adjusted_indices.reserve(coll.indices.size());
  for (std::size_t i = 0, n = coll.indices.size(); i < n; ++i) {
    const std::size_t tensor_index = coll.indices[i];
    const XLATensor &tensor = (*tensors)[tensor_index];
    bool is_restricting;
    if (IsAllowedOutput(tensor, coll, &is_restricting)) {
      adjusted_indices.push_back(coll.indices[i]);
      if (is_restricting && verbose_output_control) {
        ColorScope clr(Color::FG_DEFAULT);
        std::stringstream ss;
        ss << "Allowing output";
        if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
          ss << " HANDLE = " << tensor.data()->xla_data->GetOpaqueHandle();
        }
        TensorEx::print_tensor(ss.str(), tensor);
      }
    } else {
      if (is_restricting &&
          (verbose || verbose_output_control || verbose_remove_tensors)) {
        std::stringstream ss;
        ss << "Removing output";
        if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
          ss << " HANDLE = " << tensor.data()->xla_data->GetOpaqueHandle();
        }
        TensorEx::print_tensor(ss.str(), tensor);
      }
    }
  }

  if (adjusted_indices.empty() ||
      adjusted_indices.size() != coll.indices.size()) {
    coll.indices = std::move(adjusted_indices);
    return true;
  } else {
    return false;
  }
}

bool LazySentinel::IsTrainingThread(pid_t tid) {
  return true; // Logic has been generalized
}

/**
 * @brief Analyze the hashing situation and see if we can run this on the proxy
 * @param state
 * @param tensors
 * @param coll
 * @return
 * @note Adjusted hashing and executable's "adjusted hash" is like this:
 *
 *       For an un-pruned executable:
 *        "initial coll.hash after post-order" -> rehash( "initial coll.hash
 * after post-order" ) For a pruned executable: "initial coll.hash after
 * post-order" -> rehash( "coll.hash calculated after being sent back for prune"
 * )
 *
 *       An advantage of this is that if we are ever asked for the graph where
 * we donn't have to prune anything, it should still resolve to the same
 * executable's adjusted hash in PreProcessHlo
 *
 *       TODO: Optimize not doing two passes when we are in the same detected
 * state on first-pass entry One obvious approach is to set a flag when
 * post-order doesn't change after a prune (which so far has been the case),
 * although would like to make that optional so that the harder state can keep
 * being tested for now until there's a way to turn it on aand off and to have
 * tests for it to assure it didn't break.
 */
bool LazySentinel::OnHashingComplete(torch_xla::HashingState &_state,
                                     std::vector<XLATensor> *tensors,
                                     XLATensor::SyncTensorCollection &coll) {
  // This only updates something if this is one we have an executable for
  ProxyHashingState &state = *dynamic_cast<ProxyHashingState *>(&_state);
  static const absl::int128 PROXY_HASHING_VALUE =
      absl::MakeInt128(0xb8bc2f8ad48c431c, 0x872132d8a172a6d8);

  const std::size_t pass = state.pass_++;

  if (!is_in_mark_step /*|| !is_clean_step*/) {
    std::shared_ptr<ThreadCompileInfo> compile_info =
        GetCompileInfo(coll.requesting_tid);
    compile_info->mark_step_count_since_last_reset_ = 0;
    compile_info->sync_count_since_hash_change_ = INVALID_COUNT;
    return false;
  }

  if (!pass) {
    // First, see if we know about this hash already and it passed as
    // a stable-state executable.  That doesn't mean its impossible to
    // immediately do something to cause it to be switched, but we do
    // know this to have the ability to be stable
    if (ex_cache->has_executable(coll.hash)) {
      // just for optimizations ont eh second pass. Final state
      // should be deterministic with or without this flag
      state.known_executable_ = true;

      auto compile_info = GetCompileInfo(coll.requesting_tid);
      assert(compile_info->mark_step_count_since_last_reset_ !=
             INVALID_COUNT); // maybe this is ok, we just switched to a known
                             // graph?
      ++compile_info->mark_step_count_since_last_reset_; // is there any point
                                                         // to increment this?
      ex_cache->activate_hash(
          coll.hash);           // not sure if 'active' has a meaning anymore
      state.fabric_run_ = true; // <-- can this state just be a thread var?
      mark_step_was_on_proxy = true;
      if (PruneTensors(tensors, coll)) {
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        return true; // need to recalculate postorder with new inputs/outputs
      }
      return false; // Nothing removed, so keep going (on fabric)
    } else if (prune_tensors_if_outputs_set) {
      if (is_in_mark_step) {
        if (prev_step_was_on_proxy) {
          // This is a subsequent step to a proxy step, and
          // So if we have *intentionally* set the outputs since then,
          // prune the tensors as if it were a regular mark step.
          // This is because if there is a mark step later,
          // it may try to *only* pull in those pruned tensors,
          // so if that's the case, prune them again.
          // If the resultant list of tensors to update is empty, then
          // a sync can be skipped (as it would have been had we not pruned the
          // tensors in the first place)
          if (!coll.indices.empty()) {
            std::vector<size_t> save_indices = coll.indices;
            if (PruneTensors(tensors, coll)) {
              if (coll.indices.empty()) {
                state.pre_prune_hash_ = coll.hash;
                coll.hash = state.start_hash_;
                --state.pass_;
                std::cout << "Pruned outputs on unknown executable.";
                mark_step_was_on_proxy = true;
                return true; // need to recalculate postorder with new
                             // inputs/outputs
              } else {
                // We didn't prune everything, so allow the computation to
                // go forward
                coll.indices = std::move(save_indices);
              }
            }
          }
        }
      }
    }

    // Note: For trusted, we don't need to analyze anything
    std::shared_ptr<ThreadCompileInfo> compile_info =
        GetCompileInfo(coll.requesting_tid);
    if (coll.hash != compile_info->hash()) {
      if (verbose || verbose_hash) {
        ColorScope clr(Color::FG_GREEN);
        std::cout << mp() << "NEW HASH: " << compile_info->hash() << " -> "
                  << coll.hash << std::endl;
      }

      compile_info->set_hash(coll.hash);
      compile_info->mark_step_count_since_last_reset_ = 0;

      // If this isn't a zero-hash (i.e. first mark step call before loop),
      // then see if it's trusted
      if (!compile_info->hash() || !IsQualifyingStep(coll.requesting_tid)) {
        return false;
      }
    }

    if (verbose || verbose_hash) {
      ColorScope clr(Color::FG_GREEN);
      std::cout << mp()
                << "SAME HASH AS LAST TIME OR TRUSTED: " << compile_info->hash()
                << std::endl;
    }
    assert(compile_info->mark_step_count_since_last_reset_ != INVALID_COUNT);
    ++compile_info->mark_step_count_since_last_reset_;
    if (IsQualifyingStep(coll.requesting_tid)) {
      if (coll.device.ordinal == 0) {
        if (verbose) {
          ColorScope clr(Color::FG_GREEN);
          std::cout << mp() << "**** QUALIFYING: " << coll.hash << std::endl;
        } else {
          std::cout << mp() << "**** Stable graph found" << std::endl;
        }
      }
      ex_cache->activate_hash(coll.hash);
      if (PruneTensors(tensors, coll)) {
        state.fabric_run_ = true;
        assert(!state.pre_prune_hash_);
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        mark_step_was_on_proxy = true;
        return true; // need to recalculate postorder with new inputs
      }
      // Do we need to hash this differently for *our* executable
      // in case we didn't prune anything?
      const hash_t proxy_hash =
          xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);
#ifndef NDEBUG
      // This shouldn't be the adjusted hash already or something went wrong
      assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
      ex_cache->set_adjusted_hash(coll.hash, proxy_hash);
      coll.hash = proxy_hash;
      state.fabric_run_ = true;
      mark_step_was_on_proxy = true;
      return false; // Nothing removed, so keep going (on fabric)
    }
  } else {
    //
    // We sent them back to recalculate
    //
    // It's possible that with different outputs, the inputs didn't change,
    // in which case, 'coll.hash' is the same as 'state.pre_prune_hash_'
    //

    assert(state.pre_prune_hash_); // this should have been set, the hash
                                   // before the prune

    assert(state.fabric_run_); // this should have been set the first pass,
                               // or else we got here by accident
    assert(mark_step_was_on_proxy);

    const hash_t proxy_hash =
        xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);

#ifndef NDEBUG
    if (verbose) {
      std::cout << "Adjusted hash for proxy from " << coll.hash << " to "
                << proxy_hash << ", which had a pre-prune hash of "
                << state.pre_prune_hash_ << std::endl;
    }
    // This shouldn't be the adjusted hash already or something went wrong
    // Addendum: on second pass it should be here, right?
    assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
    ex_cache->set_adjusted_hash(state.pre_prune_hash_, proxy_hash);
    coll.hash = proxy_hash;
    mark_step_was_on_proxy = true;
    return false;
  }
  return false;
}

bool LazySentinel::WasMarkStepOnProxy() { return mark_step_was_on_proxy; }

std::vector<xla::ComputationClient::DataPtr>
LazySentinel::NotifyScheduleSyncTensorsGraph(
    std::vector<XLATensor> *xla_tensors,
    std::vector<xla::ComputationClient::DataPtr> tensors,
    XLATensor::SyncTensorCollection *coll,
    std::shared_ptr<xla::ComputationClient::Computation> &computation) {
  if (!is_in_mark_step) {
    // Anything outside of mark step is a reset
    if (verbose_mark_step) {
      ColorScope clr(std::cout, Color::FG_RED, false);
      std::cout << "Sync tensor request outside of MarkStep" << std::endl;
    }
    std::shared_ptr<ThreadCompileInfo> compile_info =
        GetCompileInfo(coll->requesting_tid);
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->set_hash(0);
    return std::move(tensors);
  }

  if (verbose) {
    ColorScope cs(Color::FG_CYAN);
    std::cout << "LazySentinel::NotifyScheduleSyncTensorsGraph(): "
              << coll->hash << std::endl;
  }

  assert(tensors.size() == coll->indices.size());
  if (verbose_tensor_sync) {
    std::size_t index = 0;
    std::for_each(
        tensors.begin(), tensors.end(), [coll, &index, xla_tensors](auto &t) {
          ColorScope cs(Color::FG_CYAN);
          std::cout << (index + 1) << " " << coll->hash
                    << ": SyncTensorsGraph tensor shape: " << t->shape();
          if (t->HasValue()) {
            std::cout << ", handle = " << t->GetOpaqueHandle();
          }
          std::cout << " ";
          TensorEx::print_tensor("", (*xla_tensors)[coll->indices[index++]]);
          // std::cout << std::endl;
        });
  }
  return std::move(tensors);
}

void LazySentinel::NotifyStepMarkerBegin(
    const std::string &device_str, const std::vector<std::string> &devices) {
  is_clean_step = false;
  is_in_mark_step = true;
  prev_step_was_on_proxy = mark_step_was_on_proxy;
  mark_step_was_on_proxy = false;
  XLA_COUNTER("SentinelStepMarker", 1);

  static bool registered_step_requirement = false;
  if (!registered_step_requirement) {
    XLA_VALUE_METRIC("SentinelRequiredStepsSinceReset",
                     get_number_of_required_runs_since_reset());
  }
  if (verbose) {
    ColorScope clr(Color::FG_YELLOW);
    std::cout << "*************** LazySentinel::NotifyStepMarker: device="
              << device_str << std::endl
              << std::flush;
  }
  const pid_t tid = gettid();
  std::shared_ptr<ThreadCompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t current_sync_count =
      compile_info->sync_count_since_hash_change_.load();
  if (current_sync_count == INVALID_COUNT) {
    // It's the first MarkStep, so just return (at top of training loop)
    compile_info->mark_step_count_since_last_reset_ = 0;
    compile_info->sync_count_since_hash_change_ = 0;
    if (verbose_mark_step) {
      std::cout << "Unclean or precursor step detected" << std::endl;
    }
    return;
  }
  is_clean_step = true;
  ++compile_info->mark_step_count_since_last_reset_;
  if (is_clean_step) {
    XLA_COUNTER("SentinelCleanSteps", 1);
  }
}

void LazySentinel::NotifyStepMarkerEnd() {
  assert(is_in_mark_step);

#if 1 // TURNED ON FOR HEADLESS TEST
  const pid_t tid = gettid();
  auto compile_info = GetCompileInfo(tid);
  compile_info->output_ids_.clear();
#endif

  is_in_mark_step = false;
  is_clean_step = false;
}

bool LazySentinel::IsQualifyingStep(pid_t tid /*, bool or_higher*/) {
  assert(is_in_mark_step); // shouldn't we always be? then we can just call
                           // once in MarkStep
  if (!is_in_mark_step) {
    return false;
  }
  if (!HasProxyDevices()) {
    return false;
  }
  const std::shared_ptr<ThreadCompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t mark_step_count_since_reset =
      compile_info->mark_step_count_since_last_reset_.load();
  const int steps_required = get_number_of_required_runs_since_reset();
  if (steps_required == std::numeric_limits<int>::max()) {
    // This is the case of simply being turned off/disabled
    return false;
  }
  if (is_clean_step) {
    if (!steps_required) {
      return true;
    }
    if (steps_required < 0) {
      // force on step
      const auto force_on_step = static_cast<std::size_t>(-steps_required);
      if (force_on_step == mark_step_count_since_reset) {
        return true;
      }
    }
  }
  if (!mark_step_count_since_reset) {
    // The first step is superfluous since it's the top of the dataset iterator
    // loop, before any graph is built. This also takes care of disqualifying
    // due to spurious compiles within the train loop
    return false;
  }
  bool ready;
  if (!steps_required) {
    ready = true; // always ready
  } else {
    // const bool ready = mark_step_count_since_reset - 1 == steps_required;
    // ready = mark_step_count_since_reset - 1 == steps_required;
    ready = mark_step_count_since_reset - 1 > steps_required;
  }
  if (ready) {
    assert(is_clean_step); // validate it coincides with clean step logic

    if (!xla::ProxyComputationClient::IsEnabled()) {
      return false;
    }

    if (verbose) {
      ColorScope clr(std::cout, {Color::BG_BLUE, Color::FG_YELLOW});
      std::cout << "Run ready" << std::endl << std::flush;
    }
  }
  return ready;
}

bool LazySentinel::IsInitialized() {
  return xla::ProxyComputationClient::IsInitialized();
}

/**
 * @brief Optionally control the number of outputs in case of
 *        unwanted outputs added (usually by the optimizer)
 */
void LazySentinel::SetOutputs(const std::vector<at::Tensor> &output_tensors,
                              bool append) {
  if (!HasProxyDevices()) {
    return;
  }
  const pid_t tid = gettid();
  assert(IsTrainingThread(tid));
  std::shared_ptr<ThreadCompileInfo> compile_info = GetCompileInfo(tid);
  if (!append) {
    compile_info->output_ids_.clear();
  }
  for (const at::Tensor &tensor : output_tensors) {
    XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
    const bool added =
        compile_info->output_ids_.insert(xla_tensor.data()->unique_id).second;
    assert(added);
  }
}

/**
 * @brief In the case of pruning outputs, check if the supplied
 *        tensor is one of the allowed outputs.
 */
bool LazySentinel::IsAllowedOutput(const XLATensor &tensor,
                                   XLATensor::SyncTensorCollection &coll,
                                   bool *is_restricting) {
  if (!is_clean_step || !is_in_mark_step) {
    return true;
  }

  assert(HasProxyDevices());
  assert(is_in_mark_step); // gets cleared at end of step
  assert(is_clean_step);   // otherwise, why are you asking?
  std::shared_ptr<ThreadCompileInfo> compile_info =
      GetCompileInfo(coll.requesting_tid);
  if (compile_info->output_ids_.empty()) {
    if (is_restricting) {
      *is_restricting = false;
    }
    return true;
  }
  if (is_restricting) {
    *is_restricting = true;
  }
  return compile_info->output_ids_.find(tensor.data()->unique_id) !=
         compile_info->output_ids_.end();
}

} // namespace torch_xla
