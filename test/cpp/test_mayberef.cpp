#include <gtest/gtest.h>

#include <string>

#include "tensorflow/compiler/xla/xla_client/util.h"

TEST(MaybeRefTest, BasicTest) {
  using StringRef = xla::util::MaybeRef<std::string>;
  std::string storage("String storage");
  StringRef ref_storage(storage);
  EXPECT_FALSE(ref_storage.is_stored());
  EXPECT_EQ(*ref_storage, storage);

  StringRef eff_storage(std::string("Vanishing"));
  EXPECT_TRUE(eff_storage.is_stored());
  EXPECT_EQ(*eff_storage, "Vanishing");
}
