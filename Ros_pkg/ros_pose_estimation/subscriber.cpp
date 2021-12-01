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
boost::mutex m;

using namespace cv;
using namespace std;
using namespace message_filters;


typedef pcl::PointXYZ Point2;
typedef pcl::visualization::PointCloudColorHandler<pcl::PointXYZ> ColorHandler;
typedef ColorHandler::Ptr ColorHandlerPtr;

pcl::PointCloud<Point2>::Ptr cloud_xyz1 (new pcl::PointCloud<Point2>);
pcl::PointCloud<Point2>::Ptr cloud_xyz2 (new pcl::PointCloud<Point2>);


void print(std::vector<int> const &input)
{
  for (int i = 0; i < input.size(); i++) {
    std::cout << input.at(i) << ' ';
  }
}


int viewer_count = 0;
int angle;
int v2 = 0;

const string PCD_PATH="../vision_ws/src/AI_Vision/beginner_tutorials/object_pcd/";



ros::Publisher pub;
ros::Publisher pub2;

// typedef pcl::PointXYZRGBA PointT;
// typedef pcl::PointCloud<PointT> PointCloudT;


typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

pcl::visualization::PCLVisualizer visu("Alignment2");


void
print4x4Matrix (const Eigen::Matrix4f & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}



void imageCallback(const yolov3::BoundingBoxes::ConstPtr& msg,const sensor_msgs::PointCloud2ConstPtr& cloud_blob)
{ 

  cout << "MESSAGE RECEIVED" << endl;
  pcl::console::TicToc time;
  pcl::PCDWriter writer;

  // Eigen::Matrix4f final_matrix = Eigen::Matrix4f::Identity();
  std::vector<std::string> classes;
  std::vector<int> x1;
  std::vector<int> x2;
  std::vector<int> y1;
  std::vector<int> y2;
  std::vector<int> angle;

  try
  {
    // ROS_INFO(msg->masks[0]);
    for(int i = 0 ; i < msg->bounding_boxes.size(); i++)
    {
      // ROS_INFO("%s Detected",msg->bounding_boxes[i].Class.c_str());
      // cv_ptr = cv_bridge::toCvCopy(msg->masks[i], sensor_msgs::image_encodings::BGR8);
      classes.push_back(msg->bounding_boxes[i].Class);
      x1.push_back(msg->bounding_boxes[i].xmin);
      x2.push_back(msg->bounding_boxes[i].xmax);
      y1.push_back(msg->bounding_boxes[i].ymin);
      y2.push_back(msg->bounding_boxes[i].ymax);
      angle.push_back(msg->bounding_boxes[i].angle);
    }
  }
  catch (cv_bridge::Exception& e)
  {
    printf("CV_BRIDGE ERROR");
    // ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }

  
  PointCloudNT::Ptr all_aligned (new PointCloudNT);
  PointCloudNT::Ptr scene1 (new PointCloudNT);
  PointCloudNT::Ptr temp6 (new PointCloudNT);
  PointCloudNT::Ptr scene2 (new PointCloudNT);

  pcl::fromROSMsg (*cloud_blob, *scene1);
  std::cout << "PointCloud representing the Cluster: " << scene1->points.size () << " data points." << std::endl;
  // Segment the ground
  pcl::ModelCoefficients::Ptr plane (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr    inliers_plane (new pcl::PointIndices);

  // Make room for a plane equation (ax+by+cz+d=0)
  plane->values.resize (4);

  pcl::SACSegmentation<PointNT> seg;        // Create the segmentation object
  seg.setOptimizeCoefficients (true);       // Optional
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setDistanceThreshold (0.01f);
  seg.setInputCloud (scene1);
  seg.segment (*inliers_plane, *plane);


  beginner_tutorials::Mat trans_msg;
  cout << "POSE ESTIMATION START" << endl;
  for(int i = 0; i < classes.size(); i ++)
  {
    int x_rotate = 0;
    int side_compare = 0;
    int is_back = 0;

    string sub_obj_name;
    string obj_name = classes[i];

    cout << "----------------- " << obj_name << " -----------------" << endl;

    boost::shared_ptr<std::vector<int> > myvector (new std::vector<int>);   // 10 zero-initialized elements

    int x1_temp = x1[i] - 20;
    int x2_temp = x2[i] + 20;
    int y1_temp = y1[i] - 20;
    int y2_temp = y2[i] + 20;

    if (x1_temp < 0) {
      x1_temp = 0;
    }
    if (x2_temp > 1280){
      x2_temp = 1280;
    }
    if (y1_temp < 0){
      y1_temp = 0;
    }
    if (y2_temp > 720){
      y2_temp = 720;
    }


    for (int q = y1_temp; q < y2_temp; q++){
      for(int w = x1_temp; w < x2_temp; w++){
        int temp_ind = 1280*q+w;

        if (binary_search(inliers_plane->indices.begin(), inliers_plane->indices.end(),temp_ind)){
          continue;
        }

        myvector->push_back(temp_ind);
      }     
    }

    // cout<< "WHERE "<< endl;
//     // Point clouds
    PointCloudNT::Ptr object (new PointCloudNT);
    PointCloudNT::Ptr object_sub (new PointCloudNT);
    PointCloudNT::Ptr scene (new PointCloudNT);
    PointCloudNT::Ptr scene3 (new PointCloudNT);

    // Object Point cloud Extract
    pcl::ExtractIndices<PointNT> extract;
    extract.setInputCloud (scene1);
    extract.setIndices (myvector);
    extract.setNegative (false);      // Extract the inliers
    extract.filter (*scene);    // cloud_inliers contains object point cloud

    std::cout << "PointCloud representing the Cluster: " << scene->points.size () << " data points." << std::endl;

    pcl::VoxelGrid<PointNT> grid;
    float leaf = 0.005f;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (scene);
    grid.filter (*scene);
    grid.setInputCloud (scene1);
    grid.filter (*scene2);
    std::cout << "PointCloud representing the Cluster: " << scene->points.size () << " data points." << std::endl;
     

    // Load Object PCD
    if(obj_name == "bottom_black"){

      obj_name = "bottom";

    }else if(obj_name == "bottom_white"){

      x_rotate = 1;
      obj_name = "bottom";

    }else if(obj_name == "fr_black"){


      pcl::ConvexHull<PointNT> cHull;
      PointCloudNT cHull_points;
      cHull.setComputeAreaVolume(true);
      cHull.setInputCloud(scene);
      cHull.reconstruct (cHull_points);
      double area= cHull.getTotalArea();
      cout << "AREA : "<< area << endl;

      if (area > 0.041){
        obj_name = "long";
      }else{
        obj_name = "short";
      }


    }else if(obj_name == "fr_white"){

      pcl::ConvexHull<PointNT> cHull;
      PointCloudNT cHull_points;
      cHull.setComputeAreaVolume(true);
      cHull.setInputCloud(scene);
      cHull.reconstruct (cHull_points);
      double area= cHull.getTotalArea();
      cout << "AREA : "<< area << endl;
      if (area > 0.041){
        obj_name = "long";
      }else{
        obj_name = "short";
      }
      x_rotate = 1;

    }else if(obj_name == "side_black"){

      obj_name = "right";
      side_compare = 1;
      sub_obj_name = "left";

    }else if(obj_name == "side_white"){

      obj_name = "right";
      side_compare = 1;
      x_rotate = 1;
      sub_obj_name = "left";

    }else if(obj_name =="back"){

      is_back = 1;

    }


    

    pcl::io::loadPCDFile(PCD_PATH+("%s",obj_name.c_str()) +".pcd", *object);

    if(x_rotate){
      Eigen::Matrix4f initial_x_axis = Eigen::Matrix4f::Identity();
      initial_x_axis(1,1) = -1;
      initial_x_axis(2,2) = -1;
      pcl::transformPointCloudWithNormals(*object, *object, initial_x_axis);
    }
    
    visu.addPointCloud (object, ColorHandlerT (object, 100.0, 100.0, 100.0), "scene");
    visu.spin();

    float reaf = 0.008f;
    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    grid.setLeafSize (reaf, reaf, reaf);
    grid.setInputCloud (object);
    grid.filter (*object);
    std::cout << "After object Downsampling: " << object->points.size () << " data points." << std::endl;
    std::cout << "After scene Downsampling: " << scene->points.size () << " data points." << std::endl;


    if(side_compare){
      pcl::io::loadPCDFile(PCD_PATH+"left.pcd", *object_sub);
      if(x_rotate){
        Eigen::Matrix4f initial_x_axis = Eigen::Matrix4f::Identity();
        initial_x_axis(1,1) = -1;
        initial_x_axis(2,2) = -1;
        pcl::transformPointCloudWithNormals(*object_sub, *object_sub, initial_x_axis);
      }
      grid.setLeafSize (reaf, reaf, reaf);
      grid.setInputCloud (object_sub);
      grid.filter (*object_sub);
    }


    //------------------------------ REMOVE NAN  ------------------------------

    std::cout << "Remove NAN" << std::endl;

    std::vector<int> indices1;
    pcl::removeNaNFromPointCloud(*object,*object, indices1);

    std::vector<int> indices2;
    pcl::removeNaNFromPointCloud(*object_sub,*object_sub, indices2);

    std::vector<int> indices3;
    pcl::removeNaNFromPointCloud(*scene,*scene, indices3);



    //------------------------------ SCENE OBJECT EXTRACTED ------------------------------
    
    std::cout << "MASS CENTER" << std::endl;

    pcl::MomentOfInertiaEstimation <PointNT> feature_extractor;
	  feature_extractor.setInputCloud (scene);
	  feature_extractor.compute ();
	  Eigen::Vector3f mass_center;
	  feature_extractor.getMassCenter (mass_center);



    Eigen::Matrix4f center_transformation = Eigen::Matrix4f::Identity();

    float theta = -angle[i]*3.141592/180;
    center_transformation(0,3) = mass_center(0);
    center_transformation(1,3) = mass_center(1);
    center_transformation(2,3) = mass_center(2);

    center_transformation(0,0) = cos(theta);
    center_transformation(0,1) = -sin(theta);
    center_transformation(1,0) = sin(theta);
    center_transformation(1,1) = cos(theta);





    //--------------------------- Initail Aligned ---------------------------

    std::cout << "Initial align" << std::endl;

    PointCloudNT::Ptr cloud_icp (new PointCloudNT);  // ICP output point cloud
    PointCloudNT::Ptr cloud_sub (new PointCloudNT);
    PointCloudNT::Ptr cloud_in (new PointCloudNT); 
    PointCloudNT::Ptr final (new PointCloudNT);


    Eigen::Matrix4f transformation_x_axis = Eigen::Matrix4f::Identity();
    transformation_x_axis(1,1) = -1;
    transformation_x_axis(2,2) = -1;

    Eigen::Matrix4f transformation_sub = Eigen::Matrix4f::Identity();

    float score;
    float score_sub = 100;

    std::vector<float> scores;

    //Copy PointNormal to PointXYZ
    pcl::copyPointCloud(*scene,*cloud_in);
    pcl::copyPointCloud(*object,*cloud_icp);
    if(is_back){
      pcl::copyPointCloud(*object,*cloud_sub);
      
      transformation_sub = center_transformation*transformation_x_axis;

      pcl::transformPointCloudWithNormals(*cloud_sub, *cloud_sub, transformation_sub);
    }else if(side_compare){
      pcl::copyPointCloud(*object_sub,*cloud_sub);
      transformation_sub = center_transformation;
      pcl::transformPointCloudWithNormals(*cloud_sub, *cloud_sub, transformation_sub);

    }
    // pcl::copyPointCloud(*object,*cloud_sub);

    //--------------------------- ICP ---------------------------

    std::cout << "ICP" << std::endl;

    int in_iterations = 4;  // Default number of ICP iterations
    int out_iter = 10;
    int iter_num = 0;

    time.tic ();

    Eigen::Matrix4f transformation2 = Eigen::Matrix4f::Identity();
    if(is_back or side_compare){
      pcl::IterativeClosestPoint<PointNT, PointNT> icp_sub;

      for(int i = 0; i < out_iter; i++){
        icp_sub.setMaximumIterations (in_iterations);
        icp_sub.setInputSource (cloud_sub);
        icp_sub.setInputTarget (cloud_in);
        icp_sub.setMaxCorrespondenceDistance (0.05);
        icp_sub.align (*cloud_sub);
        Eigen::Matrix4f transformation_matrix2 = icp_sub.getFinalTransformation ();
        transformation2 = transformation_matrix2*transformation2;
        score_sub = icp_sub.getFitnessScore ();
      }

    }

    cout << "SCORE_SUB : " << score_sub << endl;




    pcl::transformPointCloudWithNormals(*cloud_icp, *cloud_icp, center_transformation);

	cout<< "OBJOBJOBJOBJ" << std::endl;      

    pcl::IterativeClosestPoint<PointNT, PointNT> icp;


    Eigen::Matrix4f transformation1 = Eigen::Matrix4f::Identity();
    

    for(int i = 0; i < out_iter; i++){// The Iterative Closest Point algorithm
          
      icp.setMaximumIterations (in_iterations);
      icp.setInputSource (cloud_icp);
      icp.setInputTarget (cloud_in);
      icp.setMaxCorrespondenceDistance (0.05);
      icp.align (*cloud_icp);
      Eigen::Matrix4f transformation_matrix1= icp.getFinalTransformation ();
      transformation1 = transformation_matrix1*transformation1;
      score = icp.getFitnessScore ();
      
      iter_num++;

    }

    cout << "SCORE_MAIN : " << score << endl;

    scores.push_back(score);
    scores.push_back(score_sub);

    std::cout << "Applied " << iter_num << " ICP iteration(s) in " << time.toc () << " ms" << std::endl;  

    Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();

    int min;
    min = std::min_element(scores.begin(), scores.end())- scores.begin();

    if (min == 0)
    {

      pcl::copyPointCloud(*cloud_icp,*final);
      final_transformation = transformation1*center_transformation;
      if(x_rotate){
        final_transformation = final_transformation*transformation_x_axis;
      }

    }else if (min == 1){

      pcl::copyPointCloud(*cloud_sub,*final);
      final_transformation = transformation2*transformation_sub;
      cout << "X - axis transformed" << endl;
      if(side_compare){
        obj_name = "left";
      }
      // if(is_back){
        // final_transformation = final_transformation * transformation_x_axis;
      // }
    

    }

    cout << "MIN_INDEX : " << min << endl;

    // if(x_rotate){
      // final_transformation = final_transformation*transformation_x_axis;
    // }
    *all_aligned += *scene;
    // *all_aligned += *final;
    
    cout << "--------------- " << obj_name << " FOUND ---------------"<< endl;
    print4x4Matrix (final_transformation);


    Eigen::Matrix3f final_trans = final_transformation.block<3,3>(0,0);
    
    Eigen::Quaternion<float> quats (final_trans);
    geometry_msgs::Vector3 vec;
    geometry_msgs::Quaternion quat;
    
    pcl::io::loadPCDFile(PCD_PATH+("%s",obj_name.c_str()) +".pcd", *temp6);
    pcl::transformPointCloudWithNormals(*temp6, *temp6, final_transformation);
    vec.x = final_transformation(0,3);
    vec.y = final_transformation(1,3);
    vec.z = final_transformation(2,3);

    quat.x = quats.x();
    quat.y = quats.y();
    quat.z = quats.z();
    quat.w = quats.w();

    
    trans_msg.transition.push_back(vec);
    trans_msg.rotation.push_back(quat);
    trans_msg.class_names.push_back(obj_name);

    Eigen::Affine3f affin_form;
    affin_form.matrix() = final_transformation;

    stringstream ss;
    ss << v2;
    string str = ss.str();  

    v2++;
    

  }
  v2 = 0;
    // print4x4Matrix (final_transformation);

 //    // *all_aligned += *object;
    
 // //    // *all_aligned += *object_projected;
  // *all_aligned += *scene1;

  sensor_msgs::PointCloud2 ros_cloud;
  pcl::toROSMsg(*all_aligned, ros_cloud);
  pub2.publish (ros_cloud);
  pub.publish (trans_msg);

  
  viewer_count++;





}



int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  message_filters::Subscriber<yolov3::BoundingBoxes> image_sub(nh, "/detected_objects_in_image", 10);

  message_filters::Subscriber<sensor_msgs::PointCloud2> info_sub(nh, "/points2", 10);
  TimeSynchronizer<yolov3::BoundingBoxes, sensor_msgs::PointCloud2> sync(image_sub, info_sub, 10);
  sync.registerCallback(boost::bind(&imageCallback, _1, _2));
  // cout << "ABCDEFGHIJKLMNOPQRSTUVWXYZ" << endl;

  pub = nh.advertise<beginner_tutorials::Mat> ("output/pose_estimation", 1);
  pub2 = nh.advertise<sensor_msgs::PointCloud2>("output/obj_aligned",1);

  ros::spin();



}
