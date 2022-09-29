cmake_minimum_required(VERSION 2.8.11)

project(okvis_common)

IF(libvisensor_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  MESSAGE(STATUS "Found libvisensor. Setting HAVE_LIBVISENSOR flag.")
ENDIF()

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
add_library(${PROJECT_NAME} STATIC 
  include/okvis/FrameTypedefs.hpp
  include/okvis/Measurements.hpp
  include/okvis/Parameters.hpp
  include/okvis/Variables.hpp
  include/okvis/VioBackendInterface.hpp
  include/okvis/VioFrontendInterface.hpp
  include/okvis/VioInterface.hpp
  include/okvis/VioParametersReader.hpp
  include/okvis/threadsafe/ThreadsafeQueue.hpp
  include/swift_vio/FrameTypedefs.hpp
  include/swift_vio/YamlParser.h
  src/FrameTypedefs.cpp
  src/FullStateWithExtrinsicsAsCallback.cpp
  src/InitialNavState.cpp
  src/IoUtil.cpp
  src/MapPoint.cpp
  src/Parameters.cpp
  src/PipelineParams.cpp
  src/SwiftParameters.cpp
  src/VioInterface.cpp
  src/VioParametersReader.cpp
  src/YamlHelpers.cpp)

# and link it                     
target_link_libraries(${PROJECT_NAME}
  PUBLIC okvis_util
  PUBLIC okvis_kinematics 
  PUBLIC okvis_time 
  PUBLIC okvis_cv)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
