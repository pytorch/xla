#include "torch_xla/csrc/thread_pool.h"

#include <functional>

#include "torch_xla/csrc/runtime/sys_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace torch_xla {
namespace thread {

void Schedule(std::function<void()> fn) {
  static size_t num_threads = torch_xla::runtime::sys_util::GetEnvInt(
      "XLA_THREAD_POOL_SIZE", std::thread::hardware_concurrency());
  static tsl::thread::ThreadPool pool(tsl::Env::Default(), "pytorchxla",
                                      num_threads);
  pool.Schedule(std::move(fn));
}

}  // namespace thread
}  // namespace torch_xla
