#include "third_party/xla_client/util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <unordered_map>

namespace xla {
namespace util {

using ::testing::ElementsAre;

TEST(UtilTest, Cleanup) {
  bool notify = false;

  // Set to true.
  {
    Cleanup<bool> c([&notify](bool b) { notify = b; });
    c.SetStatus(true);
  }
  EXPECT_TRUE(notify);

  // Set to false.
  {
    Cleanup<bool> c([&notify](bool b) { notify = b; });
    c.SetStatus(false);
  }
  EXPECT_FALSE(notify);

  // Releasing the cleanup will not change the `notify` to true.
  {
    Cleanup<bool> c([&notify](bool b) { notify = b; });
    c.SetStatus(true);
    c.Release();
  }
  EXPECT_FALSE(notify);
}

TEST(UtilTest, Iota) {
  EXPECT_THAT(Iota<int16_t>(5, 0, 2), ElementsAre(0, 2, 4, 6, 8));
}

TEST(UtilTest, Range) {
  EXPECT_THAT(Range<int16_t>(0, 10, 2), ElementsAre(0, 2, 4, 6, 8));
  EXPECT_THAT(Range<int16_t>(10, 0, -2), ElementsAre(10, 8, 6, 4, 2));
}

TEST(UtilTest, ToVector) {
  EXPECT_THAT(ToVector<char>(std::string("char")),
              ElementsAre('c', 'h', 'a', 'r'));
}

TEST(UtilTest, Equal) {
  EXPECT_TRUE(Equal(std::string("this"), std::string("this")));
  EXPECT_FALSE(Equal(std::string("this"), std::string("that")));
}

TEST(UtilTest, FindOr) {
  std::unordered_map<int, int> v = {{1, 1}, {2, 2}, {3, 3}};
  EXPECT_EQ(FindOr(v, 1, 7), 1);
  EXPECT_EQ(FindOr(v, 2, 7), 2);
  EXPECT_EQ(FindOr(v, 3, 7), 3);
  EXPECT_EQ(FindOr(v, 10, 7), 7);
}

TEST(UtilTest, MapInsert) {
  std::unordered_map<int, int> v;
  EXPECT_EQ(MapInsert(&v, 1, [] { return 1; }), 1);
  EXPECT_EQ(MapInsert(&v, 1, [] { return 7; }), 1);
}

}  // namespace util
}  // namespace xla