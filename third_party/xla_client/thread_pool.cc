#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace xla_env {
namespace {

tensorflow::thread::ThreadPool* CreateThreadPool() {
  tensorflow::ThreadOptions thread_options;
  return new tensorflow::thread::ThreadPool(
      tensorflow::Env::Default(), thread_options, "XlaThreadPool",
      tensorflow::port::NumSchedulableCPUs(),
      /*low_latency_hint=*/false);
}

tensorflow::thread::ThreadPool* GetThreadPool() {
  static tensorflow::thread::ThreadPool* pool = CreateThreadPool();
  return pool;
}

}  // namespace

void ScheduleClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

}  // namespace xla_env
}  // namespace xla
