#ifndef XLA_CLIENT_THREAD_POOL_H_
#define XLA_CLIENT_THREAD_POOL_H_

#include <functional>
#include <memory>
#include <thread>

namespace xla {
namespace env {

class Completion {
 public:
  class Data;

  explicit Completion(std::shared_ptr<Data> data);

  ~Completion();

  void Wait();

 private:
  std::shared_ptr<Data> data_;
};

// Schedules a closure to be run. The closure should not block waiting for other
// events.
void ScheduleClosure(std::function<void()> closure);
Completion ScheduleClosureWithCompletion(std::function<void()> closure);

// Schedules a closure which might wait for IO or other events/conditions.
void ScheduleIoClosure(std::function<void()> closure);
Completion ScheduleIoClosureWithCompletion(std::function<void()> closure);

}  // namespace env
}  // namespace xla

#endif  // XLA_CLIENT_THREAD_POOL_H_
