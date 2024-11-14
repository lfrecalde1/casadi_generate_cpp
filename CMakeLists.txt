cmake_minimum_required(VERSION 3.8)
project(my_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

#add_compile_options(-Ofast -march=native)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -Ofast")

# Add the executable for your main application
add_executable(pointers src/pointers.cpp)
add_subdirectory(matplotplusplus)

# Link the shared library by specifying its full path
target_link_libraries(pointers ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmy_function_cost_Ofast.so ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmy_function_gradient_Ofast.so ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmy_function_hessian_Ofast.so ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmy_function_delta_Ofast.so matplot)


# Include directories
target_include_directories(pointers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(pointers PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# Specify dependencies for the target
ament_target_dependencies(
  pointers
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
  "Eigen3"
)

# Install the executable, shared libraries, and headers
install(TARGETS pointers
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY lib/
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()