#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
// #include <pcl/visualization/cloud_viewer.h>

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPLYFile<pcl::PointXYZ> ("/home/jialin/Documents/VSC_Projects/readPLY/000002_000125.ply", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file trail.ply \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
  // for (size_t i = 0; i < cloud->points.size (); ++i)
  for (size_t i = 0; i < 15; ++i)
    std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z << std::endl;

  // pcl::visualization::CloudViewer viewer("pcd viewer");
	// viewer.showCloud(cloud);
	system("pause");
	return (0);
}
