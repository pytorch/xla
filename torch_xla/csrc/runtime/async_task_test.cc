#include "torch_xla/csrc/runtime/async_task.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace torch_xla {
namespace runtime {

TEST(AsyncTaskTest, BaseTest) {
  auto taskfn = []() -> int { return 17; };

  torch_xla::runtime::util::AsyncTask<int> async(std::move(taskfn));
  async.Schedule();
  async.Wait();
  EXPECT_EQ(async.GetValue(), 17);
}

TEST(AsyncTaskTest, ExceptionTest) {
  auto taskfn = []() -> int { throw std::runtime_error("Task Exception"); };

  torch_xla::runtime::util::AsyncTask<int> async(std::move(taskfn));
  async.Schedule();
  bool got_exception = false;
  try {
    async.Wait();
  } catch (const std::exception&) {
    got_exception = true;
  }
  EXPECT_TRUE(got_exception);
}

TEST(AsyncTaskTest, NoResultCopyTest) {
  struct Result {
    Result(int* counter) : counter(counter) {}
    Result(const Result& ref) : counter(ref.counter) { ++(*counter); }

    Result& operator=(const Result& ref) {
      if (this != &ref) {
        counter = ref.counter;
        ++(*counter);
      }
      return *this;
    }

    Result(Result&&) = default;
    Result& operator=(Result&&) = default;

    int* counter = nullptr;
  };

  int copy_counter = 0;
  auto taskfn = [&]() -> Result { return Result(&copy_counter); };

  torch_xla::runtime::util::AsyncTask<Result> async(std::move(taskfn));
  async.Schedule();
  async.Wait();

  Result result = async.ConsumeValue();
  EXPECT_EQ(copy_counter, 0);
  EXPECT_EQ(result.counter, &copy_counter);
}

}  // namespace runtime
}  // namespace torch_xla
