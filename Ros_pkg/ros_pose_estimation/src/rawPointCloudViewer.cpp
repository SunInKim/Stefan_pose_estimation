#include <ros/ros.h>
// #include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/Image.h"
#include <sensor_msgs/PointCloud2.h>
#include <vector>
// #include <typeinfo>
#include "yolov3/BoundingBoxes.h"
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <Eigen/Core>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/point_cloud.h>
#include "pcl/common/transformation_from_correspondences.h"
#include <pcl/filters/project_inliers.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <vtkCellArray.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <boost/thread/thread.hpp>
#include <pcl/console/time.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <std_msgs/String.h>
#include <string>
#include <iostream>  
#include <boost/thread/mutex.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/convex_hull.h>

#include <Eigen/Geometry> 
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include <beginner_tutorials/Mat.h>
#include "beginner_tutorials/PoseResult.h"

#include "robot_control_msgs/Void.h"

boost::mutex m;

using namespace cv;
using namespace std;
using namespace message_filters;


// typedef pcl::PointXYZ Point2;
typedef pcl::visualization::PointCloudColorHandler<pcl::PointXYZ> ColorHandler;
typedef ColorHandler::Ptr ColorHandlerPtr;

// pcl::PointCloud<Point2>::Ptr cloud_xyz1 (new pcl::PointCloud<Point2>);
// pcl::PointCloud<Point2>::Ptr cloud_xyz2 (new pcl::PointCloud<Point2>);

pcl::visualization::PCLVisualizer *visu;

int viewer_count = 0;
int angle;
int v2 = 0;

const string PCD_PATH="../vision_ws/src/AI_Vision/beginner_tutorials/object_pcd/";

// Pose extimation result
string request_name = "none";
string camera = "none";
string result_name = "none";
geometry_msgs::Vector3 vec;
geometry_msgs::Quaternion quat;

bool has_object = false;
bool done;


ros::Publisher pub;
ros::Publisher pub2;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

void pointCloudSubCallback(const sensor_msgs::PointCloud2 &msg);
bool viewPointCloud(robot_control_msgs::Void::Request &req, robot_control_msgs::Void::Response &res);

static sensor_msgs::PointCloud2 cloud_blob;

int main(int argc, char **argv)
{
    std::string visualFlag = "true";
    if(visualFlag == "true") visu = new pcl::visualization::PCLVisualizer("Alignment2");

    ros::init(argc, argv, "point_cloud_viewer");
    ros::NodeHandle nh;

    ros::ServiceServer service       = nh.advertiseService("viewPointCloud", viewPointCloud);
    ros::Subscriber    pointcloudSub = nh.subscribe       ("/kinectB/points2", 10, pointCloudSubCallback);

    while (ros::ok()) {
        ros::spinOnce();
    }
}

void pointCloudSubCallback(const sensor_msgs::PointCloud2 &msg) {
    cloud_blob = msg;
}

bool viewPointCloud(robot_control_msgs::Void::Request &req, robot_control_msgs::Void::Response &res) {
    PointCloudNT::Ptr scene1 (new PointCloudNT);
    pcl::fromROSMsg (cloud_blob, *scene1);

    visu->addPointCloud(scene1, ColorHandlerT (scene1, 100.0, 100.0, 100.0), "rawData");
    visu->spinOnce();

    while (!visu->wasStopped()) {
        visu->spinOnce();
    }

    visu->removeAllPointClouds();
    visu->removeAllCoordinateSystems();

    return true;
}

















