#include "ros/ros.h"
#include "beginner_tutorials/PoseResult.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <std_msgs/String.h> 
#include <string>
#include <vector>
#include <cstdlib>


using namespace std;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pose_estimate_client");

  Eigen::Quaternion<float> quats;
  Eigen::Vector3f trans;

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<beginner_tutorials::PoseResult>("pose_estimate");
  beginner_tutorials::PoseResult srv;
  srv.request.req_name = argv[1];
  srv.request.prefix = argv[2];

  if (client.call(srv))
  { 
    trans(0) = srv.response.transition[0].x;
    trans(1) = srv.response.transition[0].y;
    trans(2) = srv.response.transition[0].z;

    quats.x() = srv.response.rotation[0].x;
    quats.y() = srv.response.rotation[0].y;
    quats.z() = srv.response.rotation[0].z;
    quats.w() = srv.response.rotation[0].w;

    bool is_object = srv.response.has_obj ;
    
    string obj_name = srv.response.class_name;
    if (is_object){
      cout << trans << endl;
      cout << quats.vec() << endl;
      cout << quats.w() << endl;
      cout << is_object << endl;
      cout << obj_name << endl;
      ROS_INFO("Pose estimation finished");
    }
    else{
      ROS_INFO("The object isn't found");
    }
    
  }
  else
  {
    ROS_ERROR("Failed to call pose_estimate");
    return 1;
  }

  return 0;
}
