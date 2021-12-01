#include <string>
#include <iostream>
#include <ros/ros.h>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <thread>
#include <mutex>

// PCL Library
#include <pcl/io/io.h>
#include <pcl/console/time.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/features/moment_of_inertia_estimation.h>
// ROS Library
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <yolov3/BoundingBoxes.h>
#include <beginner_tutorials/Mat.h>
#include <beginner_tutorials/PoseResult.h>

using namespace std;
using namespace message_filters;

pcl::visualization::PCLVisualizer *visu;

mutex mutexForThread;

ros::Publisher pub;
ros::Publisher pub2;

const string PCD_PATH="/home/irl-assembly/vision_ws/src/stefan_part_detection/beginner_tutorials/object_pcd/";

string request_name;
string camera;
string result_name;
geometry_msgs::Vector3 vec;
geometry_msgs::Quaternion quat;
bool has_object;
bool done;
bool visualizerFlag = false;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

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


void msg_initial()
{
  request_name = "none";
  camera = "none";
  result_name = "none";
  has_object = false;
  done = false;
  vec.x = 0;
  vec.y = 0;
  vec.z = 0;
  quat.x = 0;
  quat.y = 0;
  quat.z = 0;
  quat.w = 0;

}


bool PoseCallback(const yolov3::BoundingBoxes::ConstPtr& msg,const sensor_msgs::PointCloud2ConstPtr& cloud_blob)
{
	//------------------------------ GET YOLO V3 MESSAGE ------------------------------

	
	pcl::console::TicToc time;
	time.tic();

	cout << "MESSAGE RECEIVED" << endl;	
	vector<string> classes;
	vector<int> x1, x2, y1, y2, angle;
	try
	{
		for(int i = 0 ; i < msg->bounding_boxes.size(); i++)
    	{
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
    }

    PointCloudT::Ptr all_aligned (new PointCloudT);
	PointCloudT::Ptr scene (new PointCloudT);

	pcl::fromROSMsg (*cloud_blob, *scene);
	beginner_tutorials::Mat trans_msg;

	//------------------------------ POSE ESTIMATION START ------------------------------
	
	cout << "POSE ESTIMATION START" << endl;
	for(int i = 0; i < classes.size(); i ++)
	{
		int x_rotate = 0;
    	int side_compare = 0;
    	int is_back = 0;

    	string sub_obj_name;
    	string obj_name = classes[i];

    	cout << "----------------- " << obj_name << " -----------------" << endl;
    	
    	//------------------------------ GET BOUNDING BOX ------------------------------
     	int x1_temp = x1[i] - 20;
	    int x2_temp = x2[i] + 20;
	    int y1_temp = y1[i] - 20;
	    int y2_temp = y2[i] + 20;
	    float theta = -angle[i]*3.141592/180;

	    int index = 0;

	    x1_temp = max(x1_temp, 0);
	    x2_temp = min(x2_temp, 1280);
	    y1_temp = max(y1_temp, 0);
	    y2_temp = min(y2_temp, 720);

	    pcl::IndicesPtr bbox_index (new vector <int>);
	    for (int q = y1_temp; q < y2_temp; q++)
	    {
    		for(int w = x1_temp; w < x2_temp; w++)
    		{
    			int index = 1280*q+w;
    			bbox_index->push_back(index);
    		}     
    	}

    	//------------------------------ DEFINE SCENE & OBJECT ------------------------------
    	PointCloudT::Ptr scene_temp (new PointCloudT);
    	PointCloudT::Ptr original_scene (new PointCloudT);
    	PointCloudT::Ptr extracted_scene (new PointCloudT);

    	PointCloudT::Ptr object (new PointCloudT);
    	PointCloudT::Ptr object_sub (new PointCloudT);
    	PointCloudT::Ptr object_aligned (new PointCloudT);

	    PointCloudT::Ptr cloud_icp (new PointCloudT);  // ICP output point cloud
	    PointCloudT::Ptr cloud_sub (new PointCloudT);
	    PointCloudT::Ptr cloud_in (new PointCloudT); 

    	//------------------------------ EXTRACT OBJECT IN SCENE ------------------------------
	
	    pcl::ExtractIndices<PointT> extract;
	    extract.setInputCloud (scene);
	    extract.setIndices (bbox_index);
	    extract.setNegative (false);      // Extract the inliers
	    extract.filter (*scene_temp);

	    // Voxel grid filter
	    pcl::VoxelGrid<PointT> grid;
        float leaf = 0.004f;
	    grid.setLeafSize (leaf, leaf, leaf);
	    grid.setInputCloud (scene_temp);
	    grid.filter (*scene_temp);

	    float lea = 0.005f;
	    grid.setLeafSize (lea, lea, lea);
	    grid.setInputCloud (scene);
	    grid.filter (*original_scene);

	    // Segmentation by color
	    PointCloudT::Ptr cloud_cluster (new PointCloudT);
	    for (int pit = 0; pit < scene_temp->points.size(); ++pit)
	    {
			int r = scene_temp->points[pit].r;
			int g = scene_temp->points[pit].g;
			int b = scene_temp->points[pit].b;

			if (not(r < 130 and g > 100 and b < 100))
			{	
				extracted_scene->points.push_back (scene_temp->points[pit]);
			}
		}

  		extracted_scene->width = extracted_scene->points.size ();
	    extracted_scene->height = 1;
	    extracted_scene->is_dense = true;
	    cout << "PointCloud representing the Scene: " << extracted_scene->points.size () << " data points." << endl;

	    //------------------------------ LOAD OBJECT PCD------------------------------
	  
	    if(obj_name == "bottom_black")
	    {
	    	obj_name = "bottom";
	    	x_rotate = 1;

	    }else if(obj_name == "bottom_white")
	    {
	    	obj_name = "bottom";

	    }else if(obj_name == "long_black")
	    {			
		    obj_name = "long";
			x_rotate = 1;

	    }else if(obj_name == "long_white")
	    {
			obj_name = "long";

	    }else if(obj_name == "short_black")
	    {
		    obj_name = "short";
			x_rotate = 1;

	    }else if(obj_name == "short_white")
	    {
		    obj_name = "short";

	    }else if(obj_name == "side_black")
	    {
	    	obj_name = "right";
	    	side_compare = 1;
	    	x_rotate = 1;
	    	sub_obj_name = "left";

	    }else if(obj_name == "side_white")
	    {
	    	obj_name = "right";
	    	side_compare = 1;	    	
	    	sub_obj_name = "left";

	    }else if(obj_name =="back")
	    {
	    	is_back = 1;
	    }

	    // Determining the existence of an object
	    if(obj_name != request_name and sub_obj_name != request_name)
	    {
	    	has_object = false;
    		done = true;
    		continue;
	    }
	    pcl::io::loadPCDFile(PCD_PATH+("%s",obj_name.c_str()) +".pcd", *object);
	    Eigen::Matrix4f x_axis = Eigen::Matrix4f::Identity();
	    x_axis(1,1) = -1;
	    x_axis(2,2) = -1;

	    if(x_rotate){	    	
	    	pcl::transformPointCloud(*object, *object, x_axis);
	    }

	    // Object Downsampling
	    pcl::console::print_highlight ("Downsampling...\n");
	    grid.setLeafSize (leaf, leaf, leaf);
	    grid.setInputCloud (object);
	    grid.filter (*object);   

	    if(side_compare){
	    	pcl::io::loadPCDFile(PCD_PATH+"left.pcd", *object_sub);
	    	if(x_rotate){
	        	pcl::transformPointCloud(*object_sub, *object_sub, x_axis);
	    	}
	    	grid.setLeafSize (leaf, leaf, leaf);
	    	grid.setInputCloud (object_sub);
	    	grid.filter (*object_sub);
	    }
	    cout << "After object Downsampling: " << object->points.size () << " data points." << endl;

	    //--------------------------- Initail Aligned --------------------------------
	   
	    pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
		feature_extractor.setInputCloud (extracted_scene);
		feature_extractor.compute ();
		Eigen::Vector3f mass_center;
		feature_extractor.getMassCenter (mass_center);

	    Eigen::Matrix4f center_transformation = Eigen::Matrix4f::Identity();
	    
	    center_transformation(0,3) = mass_center(0);
	    center_transformation(1,3) = mass_center(1);
	    center_transformation(2,3) = mass_center(2);

	    center_transformation(0,0) = cos(theta);
	    center_transformation(0,1) = -sin(theta);
	    center_transformation(1,0) = sin(theta);
	    center_transformation(1,1) = cos(theta);

	    pcl::copyPointCloud(*extracted_scene,*cloud_in);
	    pcl::copyPointCloud(*object,*cloud_icp);
	    Eigen::Matrix4f transformation_sub = Eigen::Matrix4f::Identity();
	    if(is_back){
	      pcl::copyPointCloud(*object,*cloud_sub);	      
	      transformation_sub = center_transformation*x_axis;
	      pcl::transformPointCloud(*cloud_sub, *cloud_sub, transformation_sub);
	    }else if(side_compare){
	      pcl::copyPointCloud(*object_sub,*cloud_sub);
	      transformation_sub = center_transformation;
	      pcl::transformPointCloud(*cloud_sub, *cloud_sub, transformation_sub);
	    }

	    cout << "Initial Aligned" << endl;

	    //--------------------------- ITERATIVE CLOSEST POINTS ---------------------------
	   
	    int in_iterations = 4;  // Default number of ICP iterations
        int out_iter = 10;    

        float score;
	    float score_sub = 100;
	    vector<float> scores;

	    // The Iterative Closest Point algorithm for main object
	   
	    pcl::transformPointCloud(*cloud_icp, *cloud_icp, center_transformation);
	    Eigen::Matrix4f transformation1 = Eigen::Matrix4f::Identity();
	    pcl::IterativeClosestPoint<PointT, PointT> icp;
	    for(int i = 0; i < out_iter; i++)  
	    {	          
		    icp.setMaximumIterations (in_iterations);
		    icp.setInputSource (cloud_icp);
		    icp.setInputTarget (cloud_in);
		    icp.setMaxCorrespondenceDistance (0.05);
		    icp.align (*cloud_icp);
		    Eigen::Matrix4f transformation_matrix1= icp.getFinalTransformation ();
		    transformation1 = transformation_matrix1*transformation1;
		    score = icp.getFitnessScore ();		      
	    }
	    cout << "SCORE_MAIN : " << score << endl;

	    // The Iterative Closest Point algorithm for sub object
	    Eigen::Matrix4f transformation2 = Eigen::Matrix4f::Identity();
	    if(is_back or side_compare)
	    {
	    	pcl::IterativeClosestPoint<PointT, PointT> icp_sub;
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

	    int min;
	    scores.push_back(score);
	    scores.push_back(score_sub);
	    min = min_element(scores.begin(), scores.end())- scores.begin();

	    Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();

	    if (min == 0)
	    {
	    	pcl::copyPointCloud(*cloud_icp, *object_aligned);
	    	final_transformation = transformation1*center_transformation;
	    	if(x_rotate){
	        	final_transformation = final_transformation*x_axis;
	    	}

	    }else if (min == 1){

		    pcl::copyPointCloud(*cloud_sub, *object_aligned);
		    final_transformation = transformation2*transformation_sub;
		    cout << "X - axis transformed" << endl;
		    if(side_compare){
		        obj_name = "left";
	    	}
    	}

    	cout << "MIN_INDEX : " << min << endl;

    	*all_aligned += *object_aligned;
	    
	    has_object = true;
	    result_name = obj_name;

	    if(obj_name != request_name){
	    	has_object = false;
    		done = true;
    		continue;
	    }
	    cout << "--------------- " << obj_name << " FOUND ---------------"<< endl;

	    //--------------------------- GENERATE ROS MESSAGE ---------------------------
	   
	    print4x4Matrix (final_transformation);  
	    pcl::io::loadPCDFile(PCD_PATH+("%s",obj_name.c_str()) +".pcd", *object_aligned);
	    pcl::transformPointCloud(*object_aligned, *object_aligned, final_transformation);
	    
	    Eigen::Matrix3f final_trans = final_transformation.block<3,3>(0,0);	    
	    Eigen::Quaternion<float> quats (final_trans);
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
	    
		sensor_msgs::PointCloud2 ros_cloud;
		pcl::toROSMsg(*all_aligned, ros_cloud);
		pub2.publish (ros_cloud);
		pub.publish (trans_msg);


		cout << "Applied Pose estimation(s) in " << time.toc () << " ms" << endl;  
		
        if(visu != nullptr) {
            visu->removeAllPointClouds();
            visu->removeAllCoordinateSystems();

            visu->addPointCloud (object_aligned , ColorHandlerT (object_aligned, 10.0, 100.0, 255.0), ("%s_2",obj_name.c_str()));
            visu->addPointCloud (extracted_scene, ColorHandlerT (extracted_scene, 255.0, 100.0, 255.0), "extracted_scene");
            visu->addPointCloud (original_scene , ColorHandlerT (original_scene, 100.0, 100.0, 100.0), "original_scene");
            visu->addCoordinateSystem (0.1, affin_form,string("object_coord"));
            visu->spinOnce();

            bool visualizerTrigger = false;

            thread visuCounter([&] () {
                sleep(2);
                unique_lock<mutex> ul(mutexForThread);
                visualizerTrigger = true;
            });

            while (true) {
                visu->spinOnce();
                unique_lock<mutex> ul(mutexForThread);
                if (visualizerTrigger) break;
            }

            visuCounter.join();
        }
		done = true;
        return true;
	}
}

bool pose(beginner_tutorials::PoseResult::Request &req, beginner_tutorials::PoseResult::Response &res)
{
	ros::NodeHandle nh;
	request_name = req.req_name;
	camera = req.prefix;

	cout << "Requested object : " << request_name << endl;
	cout << camera << " On" << endl;

	message_filters::Subscriber<yolov3::BoundingBoxes> angle_sub(nh, "/" + camera + "/detected_objects_in_image", 10);
	message_filters::Subscriber<sensor_msgs::PointCloud2> point_sub(nh, "/" + camera+ "/points2", 10);
    TimeSynchronizer<yolov3::BoundingBoxes, sensor_msgs::PointCloud2> sync(angle_sub, point_sub, 10);
	sync.registerCallback(boost::bind(&PoseCallback, _1, _2));
 
	pub = nh.advertise<beginner_tutorials::Mat> ("output/pose_estimation", 1);
	pub2 = nh.advertise<sensor_msgs::PointCloud2>("output/obj_aligned",1);
  
	while (not done){
		ros::spinOnce();
	}
	res.transition.push_back(vec);
	res.rotation.push_back(quat);
	res.has_obj = has_object;
	res.class_name = result_name;

	if(done){
    	cout << "-----------------------DONE-----------------------" << endl;
    	msg_initial();
    	return true;
	}
}

int main(int argc, char **argv) {
    string visualFlag = argv[1];

    if(visualFlag == "true") visu = new pcl::visualization::PCLVisualizer("Alignment2");

    ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh2;
	cout << "Ready to pose estimate" << endl;
	ros::ServiceServer service = nh2.advertiseService("pose_estimate", pose);

    ros::Rate loop_rate(100);

    while (ros::ok()) {
        loop_rate.sleep();
        ros::spinOnce();
    }
}
