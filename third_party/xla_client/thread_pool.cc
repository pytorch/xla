#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace xla_env {
namespace {

tensorflow::thread::ThreadPool* CreateThreadPool(const char* name,
                                                 const char* cfg_env) {
  int64 num_threads =
      sys_util::GetEnvInt(cfg_env, tensorflow::port::NumSchedulableCPUs());
  tensorflow::ThreadOptions thread_options;
  return new tensorflow::thread::ThreadPool(
      tensorflow::Env::Default(), thread_options, name, num_threads,
      /*low_latency_hint=*/false);
}

tensorflow::thread::ThreadPool* GetThreadPool() {
  static tensorflow::thread::ThreadPool* pool =
      CreateThreadPool("XlaThreadPool", "XLA_THREAD_POOL_SIZE");
  return pool;
}

tensorflow::thread::ThreadPool* GetIoThreadPool() {
  static tensorflow::thread::ThreadPool* pool =
      CreateThreadPool("XlaIoThreadPool", "XLA_IO_THREAD_POOL_SIZE");
  return pool;
}

}  // namespace

void ScheduleClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}

}  // namespace xla_env
}  // namespace xla
