#ifndef IO_WRAP_YAMLHELPERS_H
#define IO_WRAP_YAMLHELPERS_H

#include <Eigen/Core>

namespace swift_vio {
class YamlHelpers
{
public:

  static void writeMatToYaml(const Eigen::MatrixXd &mat, const std::string &key,
                             std::ofstream &stream, const std::string pad);
};
}  // namespace swift_vio
#endif // IO_WRAP_YAMLHELPERS_H
