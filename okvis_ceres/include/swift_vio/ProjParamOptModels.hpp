#ifndef INCLUDE_SWIFT_VIO_PROJ_PARAM_OPT_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_PROJ_PARAM_OPT_MODELS_HPP_

#include <vector>
#include <Eigen/Core>
#include <okvis/ModelSwitch.hpp>

namespace swift_vio {
class ProjectionOptFXY_CXY {
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

class ProjectionOptFX_CXY {
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

class ProjectionOptFX {
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

#ifndef PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASES                \
  PROJ_OPT_MODEL_CASE(ProjectionOptFXY_CXY) \
  PROJ_OPT_MODEL_CASE(ProjectionOptFX_CXY)  \
  PROJ_OPT_MODEL_CASE(ProjectionOptFX)
#endif

inline int ProjectionOptGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ProjectionOptNameToId(std::string pinhole_opt_rep, bool* isFixed=nullptr) {
  std::transform(pinhole_opt_rep.begin(), pinhole_opt_rep.end(),
                 pinhole_opt_rep.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (isFixed) {
      *isFixed = false;
  }
  if (pinhole_opt_rep.compare("FXY_CXY") == 0) {
    return ProjectionOptFXY_CXY::kModelId;
  } else if (pinhole_opt_rep.compare("FX_CXY") == 0) {
    return ProjectionOptFX_CXY::kModelId;
  } else if (pinhole_opt_rep.compare("FX") == 0) {
    return ProjectionOptFX::kModelId;
  } else {
    if (isFixed) {
        *isFixed = true;
    }
    return ProjectionOptFXY_CXY::kModelId;
  }
}

inline std::string ProjectionModelIdToName(int model_id) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::kName;

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
  return "";
}

inline void ProjectionOptMinimalIntrinsicJacobian(
    int model_id, Eigen::Matrix2Xd* intrinsicJac) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::minimalIntrinsicJacobian(intrinsicJac);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

// apply estimated projection intrinsic params to the full param vector
// be careful global_proj_params may contain trailing distortion params
inline void ProjectionOptLocalToGlobal(int model_id,
                                       const Eigen::VectorXd& local_opt_params,
                                       Eigen::VectorXd* global_proj_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::localToGlobal(local_opt_params, global_proj_params);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ProjectionOptGlobalToLocal(
    int model_id, const Eigen::VectorXd& global_proj_params,
    Eigen::VectorXd* local_opt_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::globalToLocal(global_proj_params, local_opt_params);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

inline Eigen::MatrixXd ProjectionModelGetInitCov(int model_id,
                                                 double sigma_focal_length,
                                                 double sigma_principal_point) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::getInitCov(sigma_focal_length, sigma_principal_point);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
  return Eigen::MatrixXd();
}

inline void ProjectionOptToDimensionLabels(int model_id, std::vector<std::string>* dimensionLabels) {
    switch (model_id) {
  #define MODEL_CASES PROJ_OPT_MODEL_CASES
  #define PROJ_OPT_MODEL_CASE(ProjOptModel) \
    case ProjOptModel::kModelId:            \
      return ProjOptModel::toDimensionLabels(dimensionLabels);

      MODEL_SWITCH_CASES

  #undef PROJ_OPT_MODEL_CASE
  #undef MODEL_CASES
    }
}

inline void ProjectionOptToDesiredStdevs(int model_id,
                                         Eigen::VectorXd *desiredStdevs) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel)                                      \
  case ProjOptModel::kModelId:                                                 \
    return ProjOptModel::toDesiredStdevs(desiredStdevs);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_PROJ_PARAM_OPT_MODELS_HPP_
