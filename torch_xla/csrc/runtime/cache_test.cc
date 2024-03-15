#include "torch_xla/csrc/runtime/cache.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <string>

namespace torch_xla {
namespace runtime {
namespace util {

TEST(UtilTest, XlaUtilCacheTest) {
  static const int kMaxSize = 64;
  torch_xla::runtime::util::Cache<int, std::string> cache(kMaxSize);

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

TEST(UtilTest, XlaUtilPersistentCacheTest) {
  static const int kMaxSize = 64;
  auto serialize_fn = [](std::shared_ptr<std::string> value) -> std::string {
    return *value;
  };
  auto deserialize_fn = [](std::string value) -> std::shared_ptr<std::string> {
    return std::make_shared<std::string>(value);
  };
  char format[] = "/tmp/tmp.XXXXXX";
  char* tmpdir = mkdtemp(format);
  std::cerr << "Made tmpdir " << std::string(tmpdir) << std::endl;
  ASSERT_NE(tmpdir, nullptr);
  auto cache = std::make_unique<PersistentCache<int, std::string>>(
      kMaxSize, std::string(tmpdir), /*readonly=*/false, serialize_fn,
      deserialize_fn);

  // Add more than kMaxSize so that the memory cache will evict some from LRU.
  for (int i = 0; i < 2 * kMaxSize; ++i) {
    std::string istr = std::to_string(i);
    auto ptr = cache->Add(i, std::make_shared<std::string>(istr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);

    ptr = cache->Get(i);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);
  }

  // Ensure that the cache is able to Get all values even though the memory
  // cache doesn't track them.
  for (int i = 0; i < 2 * kMaxSize; ++i) {
    std::string istr = std::to_string(i);
    // Read through the memory cache should miss
    auto ptr = cache->GetMemoryCache().Get(i);
    EXPECT_EQ(ptr, nullptr);

    // Read through the persistent cache should hit
    ptr = cache->Get(i);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);
  }

  // Verify erasure works.
  auto ptr = cache->Add(-1, std::make_shared<std::string>("MINUS"));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(*ptr, "MINUS");
  EXPECT_TRUE(cache->Erase(-1));
  ptr = cache->Get(-1);
  EXPECT_EQ(ptr, nullptr);

  // Test a readonly cache
  cache = std::make_unique<PersistentCache<int, std::string>>(
      kMaxSize, std::string(tmpdir), /*readonly=*/true, serialize_fn,
      deserialize_fn);

  // Add values in the range [2 * kMaxSize, 4 * kMaxSize), which are disjoint
  // from what were added by the non-readonly cache.
  for (int i = 2 * kMaxSize; i < 4 * kMaxSize; ++i) {
    std::string istr = std::to_string(i);
    auto ptr = cache->Add(i, std::make_shared<std::string>(istr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);

    // Read through the memory cache
    ptr = cache->Get(i);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);
  }

  // Values which were added by the non-readonly cache should still be
  // accessible.
  for (int i = 0; i < 2 * kMaxSize; ++i) {
    std::string istr = std::to_string(i);
    // Read through the memory cache should miss
    auto ptr = cache->GetMemoryCache().Get(i);
    EXPECT_EQ(ptr, nullptr);

    // Read through the persistent cache should hit
    ptr = cache->Get(i);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, istr);

    // Attempts to erase should fail
    ASSERT_TRUE(!cache->Erase(i));
  }

  // Any values added by the readonly cache should not be accessible, since
  // they have been evicted from the memory cache.
  for (int i = 2 * kMaxSize; i < 3 * kMaxSize; ++i) {
    ASSERT_EQ(cache->Get(i), nullptr);
  }

  // Clearing the readonly cache should not impact existing values on disk.
  cache->Clear();
  for (int i = 0; i < 2 * kMaxSize; ++i) {
    // The memory cache has been cleared, so no values are tracked.
    ASSERT_EQ(cache->GetMemoryCache().Get(i), nullptr);
    // The persistent cache will read from disk.
    ASSERT_NE(cache->Get(i), nullptr);
  }

  // Recreate in non-readonly mode for a final Clear to erase all values on
  // disk.
  cache = std::make_unique<PersistentCache<int, std::string>>(
      kMaxSize, std::string(tmpdir), /*readonly=*/false, serialize_fn,
      deserialize_fn);
  cache->Clear();
  for (int i = 0; i < 4 * kMaxSize; ++i) {
    // The memory cache has been cleared, so no values are tracked.
    ASSERT_EQ(cache->GetMemoryCache().Get(i), nullptr);
    // The persistent cache will read from disk.
    ASSERT_EQ(cache->Get(i), nullptr);
  }

  unlink(tmpdir);
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla
