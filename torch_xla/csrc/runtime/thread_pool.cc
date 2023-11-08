#include "torch_xla/csrc/runtime/thread_pool.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>

#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace torch_xla {
namespace runtime {

void Schedule(std::function<void()> fn) {
  static size_t num_threads = sys_util::GetEnvInt(
      "XLA_THREAD_POOL_SIZE", std::thread::hardware_concurrency());
  static tsl::thread::ThreadPool pool(tsl::Env::Default(), "pytorchxla", num_threads);
  pool.Schedule(std::move(fn));
}

}  // namespace runtime
}  // namespace torch_xla
