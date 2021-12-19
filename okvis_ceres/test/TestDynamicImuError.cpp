
#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <swift_vio/ceres/DynamicImuError.hpp>
#include <swift_vio/ceres/NormalVectorParameterBlock.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>

#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

class DynamicImuErrorTest {
public:
void setup() {
	// initialize random number generator
  //srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

	// Build the problem.
	// set the imu parameters
	imuParameters.a0.setZero();
	imuParameters.g = 9.81;
	imuParameters.a_max = 1000.0;
	imuParameters.g_max = 1000.0;
	imuParameters.rate = 1000; // 1 kHz
	imuParameters.sigma_g_c = 6.0e-4;
	imuParameters.sigma_a_c = 2.0e-3;
	imuParameters.sigma_gw_c = 3.0e-6;
	imuParameters.sigma_aw_c = 2.0e-5;
	imuParameters.tau = 3600.0;

	// generate random motion
	const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
	const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
	const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
	const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
	const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
	const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
	const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
	const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
	const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
	const double w_a_W_x = Eigen::internal::random(0.1,10.0);
	const double w_a_W_y = Eigen::internal::random(0.1,10.0);
	const double w_a_W_z = Eigen::internal::random(0.1,10.0);
	const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
	const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
	const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
	const double m_a_W_x = Eigen::internal::random(0.1,10.0);
	const double m_a_W_y = Eigen::internal::random(0.1,10.0);
	const double m_a_W_z = Eigen::internal::random(0.1,10.0);

	// generate randomized measurements - duration 10 seconds
	okvis::kinematics::Transformation T_WS;
	//T_WS.setRandom();

	// time increment
	const double dt=1.0/double(imuParameters.rate); // time discretization

	// states
	Eigen::Quaterniond q=T_WS.q();
	Eigen::Vector3d r=T_WS.r();
	okvis::SpeedAndBias speedAndBias;
	speedAndBias.setZero();
	Eigen::Vector3d v=speedAndBias.head<3>();

	for(size_t i=0; i<size_t(duration*imuParameters.rate); ++i){
	  double time = double(i)/imuParameters.rate;
	  if (i==10){ // set this as starting pose
		  T_WS_0 = T_WS;
		  speedAndBias_0=speedAndBias;
		  t_0=okvis::Time(time);
	  }
	  if (i==size_t(duration*imuParameters.rate)-10){ // set this as starting pose
		  T_WS_1 = T_WS;
		  speedAndBias_1=speedAndBias;
		  t_1=okvis::Time(time);
	  }

	  Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
			  m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
			  m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
	  Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
				  m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
				  m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

	  //omega_S.setZero();
	  //a_W.setZero();

	  Eigen::Quaterniond dq;

	  // propagate orientation
	  const double theta_half = omega_S.norm()*dt*0.5;
		const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
	  const double cos_theta_half = cos(theta_half);
	  dq.vec()=sinc_theta_half*0.5*dt*omega_S;
	  dq.w()=cos_theta_half;
	  q = q * dq;

	  // propagate speed
	  v+=dt*a_W;

	  // propagate position
	  r+=dt*v;

	  // T_WS
	  T_WS = okvis::kinematics::Transformation(r,q);

	  // speedAndBias - v only, obviously, since this is the Ground Truth
	  speedAndBias.head<3>()=v;

	  // generate measurements
	  Eigen::Vector3d gyr = omega_S + imuParameters.sigma_g_c/sqrt(dt)
			  *Eigen::Vector3d::Random();
	  Eigen::Vector3d acc = T_WS.inverse().C()*(a_W+Eigen::Vector3d(0,0,imuParameters.g)) + imuParameters.sigma_a_c/sqrt(dt)
				*Eigen::Vector3d::Random();
	  imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(time),okvis::ImuSensorReadings(gyr,acc)));
	}

	// create the pose parameter blocks
	okvis::kinematics::Transformation T_disturb;
	T_disturb.setRandom(1,0.02);
	T_WS_1_disturbed=T_WS_1*T_disturb;
	poseParameterBlock_0 = okvis::ceres::PoseParameterBlock(T_WS_0,0,t_0); // ground truth
	poseParameterBlock_1 = okvis::ceres::PoseParameterBlock(T_WS_1_disturbed,2,t_1); // disturbed...
	problem.AddParameterBlock(poseParameterBlock_0.parameters(),okvis::ceres::PoseParameterBlock::Dimension);
	problem.AddParameterBlock(poseParameterBlock_1.parameters(),okvis::ceres::PoseParameterBlock::Dimension);
	//problem.SetParameterBlockConstant(poseParameterBlock_0.parameters());

	// create the speed and bias
	speedAndBiasParameterBlock_0 = okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias_0,1,t_0);
	speedAndBiasParameterBlock_1 = okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias_1,3,t_1);
	problem.AddParameterBlock(speedAndBiasParameterBlock_0.parameters(),okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
	problem.AddParameterBlock(speedAndBiasParameterBlock_1.parameters(),okvis::ceres::SpeedAndBiasParameterBlock::Dimension);

	// let's use our own local quaternion perturbation
	std::cout<<"setting local parameterization for pose... "<<std::flush;
	::ceres::LocalParameterization* poseLocalParameterization2d = new okvis::ceres::PoseLocalParameterization2d;
	::ceres::LocalParameterization* poseLocalParameterization = new okvis::ceres::PoseLocalParameterization;
	problem.SetParameterization(poseParameterBlock_0.parameters(),poseLocalParameterization2d);
	problem.SetParameterization(poseParameterBlock_1.parameters(),poseLocalParameterization);

	::ceres::LocalParameterization* normalVectorParameterization = new swift_vio::NormalVectorParameterization();
	gravityDirectionBlock = okvis::ceres::NormalVectorParameterBlock(imuParameters.gravityDirection(), 4);
	problem.AddParameterBlock(gravityDirectionBlock.parameters(), okvis::ceres::NormalVectorParameterBlock::Dimension);
	problem.SetParameterization(gravityDirectionBlock.parameters(), normalVectorParameterization);
	problem.SetParameterBlockConstant(gravityDirectionBlock.parameters());
	std::cout<<" [ OK ] "<<std::endl;
}

void addImuError() {
		typedef okvis::ceres::DynamicImuError<swift_vio::Imu_BG_BA> DynamicImuErrorT;
	// create the Imu error term
	DynamicImuErrorT *cost_function_imu =
			new DynamicImuErrorT(imuMeasurements, imuParameters, t_0, t_1);
	std::vector<double *> params = {
			poseParameterBlock_0.parameters(),
			speedAndBiasParameterBlock_0.parameters(),
			poseParameterBlock_1.parameters(),
			speedAndBiasParameterBlock_1.parameters(),
			gravityDirectionBlock.parameters()};

	cost_function_imu->AddParameterBlock(7);
	cost_function_imu->AddParameterBlock(9);
	cost_function_imu->AddParameterBlock(7);
	cost_function_imu->AddParameterBlock(9);
	cost_function_imu->AddParameterBlock(3);
	cost_function_imu->SetNumResiduals(15);

	problem.AddResidualBlock(cost_function_imu, NULL, params);
	// check Jacobians: only by manual inspection...
	// they verify pretty badly due to the fact that the information matrix is also a function of the states
	double* parameters[5];
	parameters[0]=poseParameterBlock_0.parameters();
	parameters[1]=speedAndBiasParameterBlock_0.parameters();
	parameters[2]=poseParameterBlock_1.parameters();
	parameters[3]=speedAndBiasParameterBlock_1.parameters();
	parameters[4]=gravityDirectionBlock.parameters();
	cost_function_imu->checkJacobians(parameters);
}

void addImuErrorTgTsTa() {
	typedef okvis::ceres::DynamicImuError<swift_vio::Imu_BG_BA> DynamicImuErrorT;
	// create the Imu error term
	DynamicImuErrorT *cost_function_imu =
			new DynamicImuErrorT(imuMeasurements, imuParameters, t_0, t_1);
	std::vector<double *> params = {
			poseParameterBlock_0.parameters(),
			speedAndBiasParameterBlock_0.parameters(),
			poseParameterBlock_1.parameters(),
			speedAndBiasParameterBlock_1.parameters(),
			gravityDirectionBlock.parameters()};

	cost_function_imu->AddParameterBlock(7);
	cost_function_imu->AddParameterBlock(9);
	cost_function_imu->AddParameterBlock(7);
	cost_function_imu->AddParameterBlock(9);
	cost_function_imu->AddParameterBlock(3);
	cost_function_imu->SetNumResiduals(15);

	problem.AddResidualBlock(cost_function_imu, NULL, params);
	// check Jacobians: only by manual inspection...
	// they verify pretty badly due to the fact that the information matrix is also a function of the states
	double* parameters[5];
	parameters[0]=poseParameterBlock_0.parameters();
	parameters[1]=speedAndBiasParameterBlock_0.parameters();
	parameters[2]=poseParameterBlock_1.parameters();
	parameters[3]=speedAndBiasParameterBlock_1.parameters();
	parameters[4]=gravityDirectionBlock.parameters();
	cost_function_imu->checkJacobians(parameters);
}

void addPriors() {
	// let's also add some priors to check this alongside
	::ceres::CostFunction* cost_function_pose = new okvis::ceres::PoseError(T_WS_0, 1e-12, 1e-4); // pose prior...
	problem.AddResidualBlock(cost_function_pose, NULL,poseParameterBlock_0.parameters());
	::ceres::CostFunction* cost_function_speedAndBias = new okvis::ceres::SpeedAndBiasError(speedAndBias_0, 1e-12, 1e-12, 1e-12); // speed and biases prior...
	problem.AddResidualBlock(cost_function_speedAndBias, NULL,speedAndBiasParameterBlock_0.parameters());
}

void solve() {
	// Run the solver!
	std::cout<<"run the solver... "<<std::endl;
	::ceres::Solver::Options options;
	//options.check_gradients=true;
	//options.numeric_derivative_relative_step_size = 1e-6;
	//options.gradient_check_relative_precision=1e-2;
	options.minimizer_progress_to_stdout = false;
	::FLAGS_stderrthreshold=google::WARNING; // enable console warnings (Jacobian verification)
	::ceres::Solver::Summary summary;
	::ceres::Solve(options, &problem, &summary);

	// print some infos about the optimization
	//std::cout << summary.FullReport() << "\n";
	std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
			<< "optimized T_WS_1 : " << poseParameterBlock_1.estimate().T() << "\n"
			<< "correct T_WS_1 : " << T_WS_1.T() << "\n";

	// make sure it converged
	OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);
	OKVIS_ASSERT_TRUE(Exception,summary.final_cost<1e-2,"cost not reducible");
	OKVIS_ASSERT_TRUE(Exception,2*(T_WS_1.q()*poseParameterBlock_1.estimate().q().inverse()).vec().norm()<1e-2,"quaternions not close enough");
	OKVIS_ASSERT_TRUE(Exception,(T_WS_1.r()-poseParameterBlock_1.estimate().r()).norm()<0.04,"translation not close enough");
}

private:
	::ceres::Problem problem;
	const double duration = 1.0;
	okvis::ImuMeasurementDeque imuMeasurements;
	okvis::ImuParameters imuParameters;
	okvis::Time t_0;
	okvis::Time t_1;

	okvis::kinematics::Transformation T_WS_0;
	okvis::SpeedAndBias speedAndBias_0;

	okvis::kinematics::Transformation T_WS_1;
	okvis::SpeedAndBias speedAndBias_1;

	okvis::kinematics::Transformation T_WS_1_disturbed;

	okvis::ceres::PoseParameterBlock poseParameterBlock_0;
	okvis::ceres::PoseParameterBlock poseParameterBlock_1;
	okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0;
	okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1;
	okvis::ceres::NormalVectorParameterBlock gravityDirectionBlock;
};

TEST(DynamicImuErrorTestSuite, Imu_BG_BA) {
	DynamicImuErrorTest test;
	test.setup();
	test.addPriors();
	test.addImuError();
	test.solve();
}

TEST(DynamicImuErrorTestSuite, Imu_BG_BA_TG_TS_TA) {
	DynamicImuErrorTest test;
	test.setup();
	test.addPriors();
	test.addImuErrorTgTsTa();
	test.solve();
}
