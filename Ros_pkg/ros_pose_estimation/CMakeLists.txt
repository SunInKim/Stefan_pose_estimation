cmake_minimum_required(VERSION 2.8.3)
project(ros_pose_estimation)
find_package(PCL 1.9 REQUIRED)
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  std_msgs
  message_generation
  cv_bridge
  image_transport
  sensor_msgs
  geometry_msgs
  robot_control_msgs
)
find_package(OpenCV)
find_package(Boost REQUIRED COMPONENTS filesystem system thread)
## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Num.msg
  Result.msg
  Mat.msg
)


add_service_files(
  FILES
  PoseResult.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_pose_estimation
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime geometry_msgs
#  DEPENDS system_lib
)

#include_directories(
# include
#  ${catkin_INCLUDE_DIRS}
#)


include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${catkin_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_libraries(${PCL_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

#add_executable(subscriber src/subscriber.cpp)
#add_dependencies(subscriber ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
#target_link_libraries(subscriber ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pose_estimation src/pose_estimation.cpp)
add_dependencies(pose_estimation ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pose_estimation ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pose_query src/pose_query.cpp)
add_dependencies(pose_query ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pose_query ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(point_cloud_viewer src/rawPointCloudViewer.cpp)
add_dependencies(point_cloud_viewer ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(point_cloud_viewer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

