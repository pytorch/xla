#include <gtest/gtest.h>

#include <string>

#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace cpp_test {

TEST(XlaUtilCacheTest, BasicTest) {
  static const int kMaxSize = 64;
  xla::util::Cache<int, std::string> cache(kMaxSize);

  for (int i = 0; i < 2 * kMaxSize; ++i) {
    std::string istr = std::to_string(i);
    auto ptr = cache.Add(i, std::make_shared<std::string>(istr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);

    ptr = cache.Get(i);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);
  }
  for (int i = 0; i < kMaxSize - 1; ++i) {
    auto ptr = cache.Get(i);
    EXPECT_EQ(ptr, nullptr);
  }

  auto ptr = cache.Add(-1, std::make_shared<std::string>("MINUS"));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(*ptr, "MINUS");
  EXPECT_TRUE(cache.Erase(-1));
  ptr = cache.Get(-1);
  EXPECT_EQ(ptr, nullptr);
}

}  // namespace cpp_test
}  // namespace torch_xla
