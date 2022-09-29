#include "swift_vio/YamlHelpers.h"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace swift_vio {

void YamlHelpers::writeMatToYaml(const Eigen::MatrixXd &mat, const std::string &key,
                    std::ofstream &stream, const std::string pad) {
  std::string lead = "- ";
  stream << pad << key << ":\n";
  std::streamsize ss = std::cout.precision();
  if (mat.cols() == 1) {
    for (int k = 0; k < mat.rows(); ++k) {
      stream << pad << lead << std::setprecision(12) << mat(k, 0) << "\n";
    }
  } else {
    stream << std::setprecision(12);
    for (int j = 0; j < mat.rows(); j++) {
      for (int k = 0; k < mat.cols(); ++k) {
        if (k == 0) {
          stream << pad << lead << lead << mat(j, k) << "\n";
        } else {
          stream << pad << "  " << lead << mat(j, k) << "\n";
        }
      }
    }
  }
  stream << std::setprecision(ss);
}
}  // namespace swift_vio
