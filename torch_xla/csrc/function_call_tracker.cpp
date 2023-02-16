#include "torch_xla/csrc/function_call_tracker.h"

#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "torch/csrc/lazy/python/python_util.h"

namespace torch_xla {
namespace fn_tracker {
namespace {

struct TrackerContext {
  TrackerContext(std::string path, int level)
      : path(std::move(path)), level(level) {}

  std::mutex lock;
  std::string path;
  int level;
  std::unordered_set<std::string> tags;
};

TrackerContext* LoadTrackerContext() {
  std::string fntracker_file =
      xla::sys_util::GetEnvString("XLA_FNTRACKER_FILE", "");
  TrackerContext* tctx = nullptr;
  if (!fntracker_file.empty()) {
    tctx = new TrackerContext(
        std::move(fntracker_file),
        xla::sys_util::GetEnvInt("XLA_FNTRACKER_LEVEL",
                                 std::numeric_limits<int>::max()));

    std::string fn_list = xla::sys_util::GetEnvString("XLA_FNTRACKER_LIST", "");
    for (auto& fn : absl::StrSplit(fn_list, ':')) {
      if (!fn.empty()) {
        tctx->tags.insert(std::string(fn));
      }
    }
  }
  return tctx;
}

TrackerContext* GetTrackerContext() {
  static TrackerContext* tctx = LoadTrackerContext();
  return tctx;
}

void LogFunction(TrackerContext* tctx, const char* tag) {
  std::lock_guard<std::mutex> guard(tctx->lock);
  std::ofstream fn_file(tctx->path, std::ios_base::app);
  fn_file << "[TAG " << tag << " From Thread " << std::this_thread::get_id()
          << "]\n"
          << torch::lazy::GetPythonFrames() << "\nC++ Frames:\n"
          << tensorflow::CurrentStackTrace() << "\n";
}

}  // namespace

void TrackFunction(const char* tag, int level) {
  TrackerContext* tctx = GetTrackerContext();
  if (tctx != nullptr && level <= tctx->level &&
      (tctx->tags.empty() || tctx->tags.count(tag) > 0)) {
    LogFunction(tctx, tag);
  }
}

}  // namespace fn_tracker
}  // namespace torch_xla
