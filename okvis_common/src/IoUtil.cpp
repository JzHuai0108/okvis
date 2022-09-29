

#include <swift_vio/IoUtil.hpp>

namespace swift_vio {
// A better implementation is given here.
// https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
std::string removeTrailingSlash(const std::string &path) {
  std::string subpath(path);
  std::size_t slash_index = subpath.find_last_of("/\\");
  while (slash_index != std::string::npos &&
         slash_index == subpath.length() - 1) {
    subpath = subpath.substr(0, slash_index);
    slash_index = subpath.find_last_of("/\\");
  }
  return subpath;
}

}  // namespace swift_vio

