cmake_minimum_required(VERSION 2.8.11)
project(loop_closure_structs)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
add_library(${PROJECT_NAME} STATIC
  src/KeyframeForLoopDetection.cpp
  include/loop_closure/KeyframeForLoopDetection.hpp)

# and link it
target_link_libraries(${PROJECT_NAME}
  PUBLIC okvis_cv
  PUBLIC okvis_common
  PRIVATE ${GLOG_LIBRARIES})

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")

# testing
if(BUILD_TESTS)
  if(APPLE)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
  else()
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  endif(APPLE)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  add_executable(${PROJECT_TEST_NAME}
    test/test_main.cpp
    test/TestInverseTransformMultiply.cpp
  )
  target_link_libraries(${PROJECT_TEST_NAME}
    ${PROJECT_NAME}
    ${GTEST_LIBRARY}
    pthread)
  add_test(test ${PROJECT_TEST_NAME})
endif()
