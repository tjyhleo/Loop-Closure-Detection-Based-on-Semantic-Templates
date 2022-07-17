#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <opencv4/opencv2/core.hpp>
// #include <pcl/visualization/cloud_viewer.h>

int main (int argc, char** argv)
{
  std::string path="/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/data_3d_semantics/2013_05_28_drive_0007_sync/static";
  cv::Mat PLY;
  std::vector<cv::String> fn;
  cv::glob(path, fn, false);
  std::size_t count = fn.size();
  std::cout<<count<<std::endl;
  for (int i=0; i<2; i++){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    //"/home/jialin/Documents/VSC_Projects/readPLY/000002_000125.ply"
    if (pcl::io::loadPLYFile<pcl::PointXYZ> (fn[i], *cloud) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file trail.ply \n");
      return (-1);
    }
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
    // for (size_t i = 0; i < cloud->points.size (); ++i)
    for (size_t i = 0; i < 15; ++i){
      std::cout << "    " << cloud->points[i].x
                << " "    << cloud->points[i].y
                << " "    << cloud->points[i].z << std::endl;
    }
    }

  // pcl::visualization::CloudViewer viewer("pcd viewer");
	// viewer.showCloud(cloud);
	// system("pause");
	return (0);
}
