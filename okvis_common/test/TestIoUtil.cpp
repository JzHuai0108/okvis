#include <gtest/gtest.h>
#include <swift_vio/IoUtil.hpp>

TEST(IoUtil, removeTrailingSlash) {
  std::string path1 = "/a/b";
  ASSERT_EQ(path1, swift_vio::removeTrailingSlash(path1));
  std::string path2 = "/a/b/";
  ASSERT_EQ("/a/b", swift_vio::removeTrailingSlash(path2));
  std::string path3 = "/a\\b\\";
  ASSERT_EQ("/a\\b", swift_vio::removeTrailingSlash(path3));
  std::string path4 = "/a/b//";
  ASSERT_EQ("/a/b", swift_vio::removeTrailingSlash(path4));
}
