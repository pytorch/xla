#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace env {
namespace {

tensorflow::thread::ThreadPool* CreateThreadPool(const char* name,
                                                 int64 num_threads) {
  tensorflow::ThreadOptions thread_options;
  return new tensorflow::thread::ThreadPool(tensorflow::Env::Default(),
                                            thread_options, name, num_threads,
                                            /*low_latency_hint=*/false);
}

tensorflow::thread::ThreadPool* GetThreadPool() {
  static int64 num_threads = sys_util::GetEnvInt(
      "XLA_THREAD_POOL_SIZE", tensorflow::port::NumSchedulableCPUs());
  static tensorflow::thread::ThreadPool* pool =
      CreateThreadPool("XlaThreadPool", num_threads);
  return pool;
}

tensorflow::thread::ThreadPool* GetIoThreadPool() {
  static int64 num_threads = sys_util::GetEnvInt(
      "XLA_IO_THREAD_POOL_SIZE", 2 * tensorflow::port::NumSchedulableCPUs());
  static tensorflow::thread::ThreadPool* pool =
      CreateThreadPool("XlaIoThreadPool", num_threads);
  return pool;
}

}  // namespace

void ScheduleClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}

}  // namespace env
}  // namespace xla
