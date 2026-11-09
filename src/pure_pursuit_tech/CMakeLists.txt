cmake_minimum_required(VERSION 3.8)
project(pure_pursuit_tech)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)

# The logic, without ROS: the path, progress along it, the control law.
add_library(pure_pursuit_tech_core STATIC src/path.cpp src/tracker.cpp src/control.cpp)
target_compile_features(pure_pursuit_tech_core PUBLIC cxx_std_17)
target_include_directories(pure_pursuit_tech_core PUBLIC include)
set_target_properties(pure_pursuit_tech_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The ROS node: settings, safety gates, topics.
add_executable(pure_pursuit_tech_node src/pure_pursuit_tech_node.cpp)
target_link_libraries(pure_pursuit_tech_node pure_pursuit_tech_core)
ament_target_dependencies(pure_pursuit_tech_node
  rclcpp nav_msgs sensor_msgs ackermann_msgs ament_index_cpp std_msgs)

install(TARGETS pure_pursuit_tech_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY waypoints DESTINATION share/${PROJECT_NAME})
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # Node tests: start the built node on /test_only/* topics and check the
  # commands it publishes. Need ROS sourced and an isolated domain:
  #   export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  #   python3 -m pytest src/pure_pursuit_tech/test/ -v
  # Unit tests of the ROS-free core (no ROS graph needed):
  #   colcon test --packages-select pure_pursuit_tech
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core pure_pursuit_tech_core)
  # Closed-loop rejoin: the control law driving a kinematic bicycle
  ament_add_gtest(test_rejoin test/test_rejoin.cpp)
  target_link_libraries(test_rejoin pure_pursuit_tech_core)

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pure_pursuit_tech_node
    test/test_pure_pursuit_tech_node.py
    TIMEOUT 300)
  ament_add_pytest_test(test_pure_pursuit_tech_launch
    test/test_pure_pursuit_tech_launch.py)
endif()

ament_package()
