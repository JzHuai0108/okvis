#ifndef SWIFT_VIO_IO_UTIL_HPP
#define SWIFT_VIO_IO_UTIL_HPP

#include <Eigen/Core>

namespace swift_vio {
// Space separated output format for Eigen matrices.
static const Eigen::IOFormat kSpaceInitFmt(Eigen::StreamPrecision,
                                           Eigen::DontAlignCols, " ", "\n", "",
                                           "", "", "");

}  // namespace swift_vio
#endif  // SWIFT_VIO_IO_UTIL_HPP

