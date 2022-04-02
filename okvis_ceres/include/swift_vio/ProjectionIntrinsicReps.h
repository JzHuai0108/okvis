#ifndef INCLUDE_SWIFT_VIO_PROJ_INTRINSIC_REPS_HPP_
#define INCLUDE_SWIFT_VIO_PROJ_INTRINSIC_REPS_HPP_

#include <vector>
#include <Eigen/Core>
#include <okvis/ModelSwitch.hpp>

namespace swift_vio {
/**
 * @brief Different parameterizations of the projection intrinsic parameters.
 * @deprecated These parameterizations are redundant wrt the camera geometry model,
 * and tend to scatter everywhere.
 */
class ProjIntrinsic_FXY_CXY {
 public:
  static const int kModelId = 1;
  static const size_t kNumParams = 4;
  static const std::string kName;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    global_proj_params->head<4>() = local_opt_params;
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    (*local_opt_params) = global_proj_params.head<4>();
  }

  static void minimalIntrinsicJacobian(Eigen::Matrix2Xd* /*intrinsicJacobian*/) {}
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double sigma_principal_point) {
    Eigen::MatrixXd covProjIntrinsics = Eigen::Matrix<double, 4, 4>::Identity();
    covProjIntrinsics.topLeftCorner<2, 2>() *= std::pow(sigma_focal_length, 2);
    covProjIntrinsics.bottomRightCorner<2, 2>() *=
        std::pow(sigma_principal_point, 2);
    return covProjIntrinsics;
  }
  static void toDimensionLabels(std::vector<std::string>* dimensionLabels) {
    *dimensionLabels = {"fx[pixel]", "fy", "cx", "cy"};
  }

  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(4);
    desiredStdevs->setConstant(1.0);
  }
};

class ProjIntrinsic_FX_CXY {
 public:
  static const int kModelId = 2;
  static const size_t kNumParams = 3;
  static const std::string kName;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    (*global_proj_params)(0) = local_opt_params[0];
    (*global_proj_params)(1) = local_opt_params[0];
    global_proj_params->segment<2>(2) = local_opt_params.segment<2>(1);
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    local_opt_params->resize(3, 1);
    (*local_opt_params)[0] = global_proj_params[0];
    local_opt_params->segment<2>(1) = global_proj_params.segment<2>(2);
  }
  static void minimalIntrinsicJacobian(Eigen::Matrix2Xd* intrinsicJac) {
    const int resultCols = intrinsicJac->cols() - 1;
    intrinsicJac->col(0) += intrinsicJac->col(1);
    intrinsicJac->block(0, 1, 2, resultCols - 1) =
        intrinsicJac->block(0, 2, 2, resultCols - 1);
    intrinsicJac->conservativeResize(Eigen::NoChange, resultCols);
  }
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double sigma_principal_point) {
    Eigen::MatrixXd covProjIntrinsics = Eigen::Matrix<double, 3, 3>::Identity();
    covProjIntrinsics(0, 0) *= std::pow(sigma_focal_length, 2);
    covProjIntrinsics.bottomRightCorner<2, 2>() *=
        std::pow(sigma_principal_point, 2);
    return covProjIntrinsics;
  }
  static void toDimensionLabels(std::vector<std::string>* dimensionLabels) {
    *dimensionLabels = {"fx[pixel]", "cx", "cy"};
  }

  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(3);
    desiredStdevs->setConstant(1.0);
  }
};

class ProjIntrinsic_FX {
 public:
  static const int kModelId = 3;
  static const size_t kNumParams = 1;
  static const std::string kName;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    (*global_proj_params)[0] = local_opt_params[0];
    (*global_proj_params)[1] = local_opt_params[0];
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    local_opt_params->resize(1, 1);
    (*local_opt_params)[0] = global_proj_params[0];
  }
  static void minimalIntrinsicJacobian(Eigen::Matrix2Xd* intrinsicJac) {
    const int resultCols = intrinsicJac->cols() - 3;
    intrinsicJac->col(0) += intrinsicJac->col(1);
    intrinsicJac->block(0, 1, 2, resultCols - 1) =
        intrinsicJac->block(0, 4, 2, resultCols - 1);
    intrinsicJac->conservativeResize(Eigen::NoChange, resultCols);
  }
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double /*sigma_principal_point*/) {
    Eigen::MatrixXd covProjIntrinsics =
        Eigen::Matrix<double, 1, 1>::Identity() *
        std::pow(sigma_focal_length, 2);
    return covProjIntrinsics;
  }
  static void toDimensionLabels(std::vector<std::string>* dimensionLabels) {
    *dimensionLabels = {"fx[pixel]"};
  }
  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(1);
    desiredStdevs->setConstant(1.0);
  }
};

#ifndef PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASES                \
  PROJ_INTRINSIC_REP_CASE(ProjIntrinsic_FXY_CXY) \
  PROJ_INTRINSIC_REP_CASE(ProjIntrinsic_FX_CXY)  \
  PROJ_INTRINSIC_REP_CASE(ProjIntrinsic_FX)
#endif

inline int ProjIntrinsicRepGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::getMinimalDim();

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ProjIntrinsicRepNameToId(std::string rep_name, bool* isFixed=nullptr) {
  std::transform(rep_name.begin(), rep_name.end(),
                 rep_name.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (isFixed) {
      *isFixed = false;
  }
  if (rep_name.compare("FXY_CXY") == 0) {
    return ProjIntrinsic_FXY_CXY::kModelId;
  } else if (rep_name.compare("FX_CXY") == 0) {
    return ProjIntrinsic_FX_CXY::kModelId;
  } else if (rep_name.compare("FX") == 0) {
    return ProjIntrinsic_FX::kModelId;
  } else {
    if (isFixed) {
        *isFixed = true;
    }
    return ProjIntrinsic_FXY_CXY::kModelId;
  }
}

inline std::string ProjectionIntrinsicRepIdToName(int model_id) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::kName;

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
  return "";
}

inline void ProjIntrinsicRepMinimalIntrinsicJacobian(
    int model_id, Eigen::Matrix2Xd* intrinsicJac) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::minimalIntrinsicJacobian(intrinsicJac);

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
}

// apply estimated projection intrinsic params to the full param vector
// be careful global_proj_params may contain trailing distortion params
inline void ProjIntrinsicRepLocalToGlobal(int model_id,
                                       const Eigen::VectorXd& local_opt_params,
                                       Eigen::VectorXd* global_proj_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::localToGlobal(local_opt_params, global_proj_params);

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
}

inline void ProjIntrinsicRepGlobalToLocal(
    int model_id, const Eigen::VectorXd& global_proj_params,
    Eigen::VectorXd* local_opt_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::globalToLocal(global_proj_params, local_opt_params);

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
}

inline Eigen::MatrixXd ProjectionIntrinsicRepGetInitCov(int model_id,
                                                 double sigma_focal_length,
                                                 double sigma_principal_point) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
  case ProjectionIntrinsicRep::kModelId:            \
    return ProjectionIntrinsicRep::getInitCov(sigma_focal_length, sigma_principal_point);

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
  return Eigen::MatrixXd();
}

inline void ProjIntrinsicRepToDimensionLabels(int model_id, std::vector<std::string>* dimensionLabels) {
    switch (model_id) {
  #define MODEL_CASES PROJ_INTRINSIC_REP_CASES
  #define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep) \
    case ProjectionIntrinsicRep::kModelId:            \
      return ProjectionIntrinsicRep::toDimensionLabels(dimensionLabels);

      MODEL_SWITCH_CASES

  #undef PROJ_INTRINSIC_REP_CASE
  #undef MODEL_CASES
    }
}

inline void ProjIntrinsicRepToDesiredStdevs(int model_id,
                                         Eigen::VectorXd *desiredStdevs) {
  switch (model_id) {
#define MODEL_CASES PROJ_INTRINSIC_REP_CASES
#define PROJ_INTRINSIC_REP_CASE(ProjectionIntrinsicRep)                                      \
  case ProjectionIntrinsicRep::kModelId:                                                 \
    return ProjectionIntrinsicRep::toDesiredStdevs(desiredStdevs);

    MODEL_SWITCH_CASES

#undef PROJ_INTRINSIC_REP_CASE
#undef MODEL_CASES
  }
}

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_PROJ_INTRINSIC_REPS_HPP_
