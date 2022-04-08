#ifndef CAMERA_MODEL_SWITCH_HPP
#define CAMERA_MODEL_SWITCH_HPP

#include <okvis/ModelSwitch.hpp>

#ifndef DISTORTION_MODEL_COMMON_SWITCH_CASES
#define DISTORTION_MODEL_COMMON_SWITCH_CASES                                   \
  case okvis::cameras::DistortionType::Equidistant:                             \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>)  \
    break;                                                                     \
  case okvis::cameras::DistortionType::RadialTangential:                        \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion>)         \
    break;                                                                     \
  case okvis::cameras::DistortionType::RadialTangential8:                       \
    DISTORTION_MODEL_CASE(okvis::cameras::PinholeCamera<                       \
                          okvis::cameras::RadialTangentialDistortion8>)        \
    break;                                                                     \
  case okvis::cameras::DistortionType::Fov:                                     \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>)          \
    break;                                                                     \
  case okvis::cameras::DistortionType::Eucm:                                    \
    DISTORTION_MODEL_CASE(okvis::cameras::EUCM)                                \
    break;
#endif

#ifndef DISTORTION_MODEL_NO_NODISTORTION_SWITCH_CASES
#define DISTORTION_MODEL_NO_NODISTORTION_SWITCH_CASES                          \
  DISTORTION_MODEL_COMMON_SWITCH_CASES                                         \
  default:                                                                     \
    MODEL_DOES_NOT_APPLY_EXCEPTION                                             \
    break;
#endif

#ifndef DISTORTION_MODEL_SWITCH_CASES
#define DISTORTION_MODEL_SWITCH_CASES                                          \
  DISTORTION_MODEL_COMMON_SWITCH_CASES                                         \
  case okvis::cameras::DistortionType::No:                            \
    DISTORTION_MODEL_CASE(                                                     \
        okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>)           \
    break;                                                                     \
  default:                                                                     \
    MODEL_DOES_NOT_EXIST_EXCEPTION                                             \
    break;
#endif

#endif // CAMERA_MODEL_SWITCH_HPP
