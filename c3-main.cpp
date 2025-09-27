#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

// pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono>
#include <ctime>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h> // TicToc

// ======= CONFIG =======
#define USE_NDT 1  // set to 0 to use ICP instead

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer)
{
  if (event.getKeySym() == "Right" && event.keyDown())
  {
    cs.push_back(ControlState(0, -0.02, 0));
  }
  else if (event.getKeySym() == "Left" && event.keyDown())
  {
    cs.push_back(ControlState(0, 0.02, 0));
  }
  if (event.getKeySym() == "Up" && event.keyDown())
  {
    cs.push_back(ControlState(0.1, 0, 0));
  }
  else if (event.getKeySym() == "Down" && event.keyDown())
  {
    cs.push_back(ControlState(-0.1, 0, 0));
  }
  if (event.getKeySym() == "a" && event.keyDown())
  {
    refresh_view = true;
  }
}

void Accuate(ControlState response, cc::Vehicle::Control &state)
{
  if (response.t > 0)
  {
    if (!state.reverse)
    {
      state.throttle = min(state.throttle + response.t, 1.0f);
    }
    else
    {
      state.reverse = false;
      state.throttle = min(response.t, 1.0f);
    }
  }
  else if (response.t < 0)
  {
    response.t = -response.t;
    if (state.reverse)
    {
      state.throttle = min(state.throttle + response.t, 1.0f);
    }
    else
    {
      state.reverse = true;
      state.throttle = min(response.t, 1.0f);
    }
  }
  state.steer = min(max(state.steer + response.s, -1.0f), 1.0f);
  state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  BoxQ box;
  box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
  box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
  box.cube_length = 4;
  box.cube_width = 2;
  box.cube_height = 2;
  renderBox(viewer, box, num, color, alpha);
}

#if USE_NDT == 1
Eigen::Matrix4d NDT(PointCloudT::Ptr mapCloud, PointCloudT::Ptr source, Pose startPose, int iterations)
{
  Eigen::Matrix4f init_guess = transform3D(startPose.rotation.yaw, startPose.rotation.pitch, startPose.rotation.roll,
                                           startPose.position.x, startPose.position.y, startPose.position.z)
                                   .cast<float>();

  pcl::NormalDistributionsTransform<PointT, PointT> ndt;
  ndt.setMaximumIterations(iterations);
  ndt.setTransformationEpsilon(1e-3);
  ndt.setResolution(5.0); // tweak as needed
  ndt.setInputSource(source);
  ndt.setInputTarget(mapCloud);

  PointCloudT::Ptr ndt_cloud(new PointCloudT);
  ndt.align(*ndt_cloud, init_guess);

  return ndt.getFinalTransformation().cast<double>();
}
#else
Eigen::Matrix4d ICP(pcl::PointCloud<PointT>::Ptr target, pcl::PointCloud<PointT>::Ptr source, Pose startPose, int iterations)
{
  Eigen::Matrix4d initTransform = transform3D(startPose.rotation.yaw, startPose.rotation.pitch, startPose.rotation.roll,
                                              startPose.position.x, startPose.position.y, startPose.position.z);
  PointCloudT::Ptr transform_source(new PointCloudT);
  pcl::transformPointCloud(*source, *transform_source, initTransform);

  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputSource(transform_source);
  icp.setInputTarget(target);
  icp.setMaximumIterations(iterations);
  icp.setMaxCorrespondenceDistance(2.0);

  PointCloudT::Ptr cloud_icp(new PointCloudT);
  icp.align(*cloud_icp);

  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity(4, 4);

  if (icp.hasConverged())
  {
    transformation_matrix = icp.getFinalTransformation().cast<double>();
    transformation_matrix = transformation_matrix * initTransform;
    return transformation_matrix;
  }
  else
  {
    cout << "warning! ICP has not converged!" << endl;
    return transformation_matrix;
  }
}
#endif

int main()
{
  auto client = cc::Client("localhost", 2000);
  client.SetTimeout(2s);
  auto world = client.GetWorld();

  auto blueprint_library = world.GetBlueprintLibrary();
  auto vehicles = blueprint_library->Filter("vehicle");

  auto map = world.GetMap();
  auto transform = map->GetRecommendedSpawnPoints()[1];
  auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

  // Create lidar
  auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
  // You can modify lidar values to get different scan resolutions
  lidar_bp.SetAttribute("upper_fov", "15");
  lidar_bp.SetAttribute("lower_fov", "-25");
  lidar_bp.SetAttribute("channels", "32");
  lidar_bp.SetAttribute("range", "30");
  lidar_bp.SetAttribute("rotation_frequency", "60");
  lidar_bp.SetAttribute("points_per_second", "500000");

  auto user_offset = cg::Location(0, 0, 0);
  auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
  auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
  auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);

  bool new_scan = true;
  std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&viewer);

  auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);

  // ===== Initial control: keep vehicle stationary until keys pressed
  control = {};
  control.brake = 1.0f;           // hold brakes initially
  control.hand_brake = false;     // you can set true if you prefer
  control.throttle = 0.0f;
  control.steer = 0.0f;
  control.reverse = false;
  vehicle->ApplyControl(control);

  // Ground-truth pose at start (world) and use it as both truth-reference and initial guess for SLAM
  Pose poseRef(
      Point(vehicle->GetTransform().location.x,
            vehicle->GetTransform().location.y,
            vehicle->GetTransform().location.z),
      Rotate(vehicle->GetTransform().rotation.yaw * pi / 180,
             vehicle->GetTransform().rotation.pitch * pi / 180,
             vehicle->GetTransform().rotation.roll * pi / 180));

  // Estimated pose (in world). Initialize with the same as ground-truth to give NDT/ICP a good start.
  Pose pose = poseRef;

  // A reference for the estimated pose to compute relative frame (like poseRef for truth)
  static bool first_est = true;
  static Pose estRef;

  // Load map
  PointCloudT::Ptr mapCloud(new PointCloudT);
  if (pcl::io::loadPCDFile("map.pcd", *mapCloud) != 0)
  {
    cerr << "Failed to load map.pcd" << endl;
    return 1;
  }
  cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
  renderPointCloud(viewer, mapCloud, "map", Color(0, 0, 1));

  typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr scanCloud(new pcl::PointCloud<PointT>);

  lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data)
                {
                  if (new_scan)
                  {
                    auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
                    for (auto detection : *scan)
                    {
                      // Don't include points too close to ego
                      if ((detection.point.x * detection.point.x + detection.point.y * detection.point.y + detection.point.z * detection.point.z) > 8.0)
                      {
                        pclCloud.points.push_back(PointT(detection.point.x, detection.point.y, detection.point.z));
                      }
                    }
                    if (pclCloud.points.size() > 5000) // adjust for desired scan density
                    {
                      lastScanTime = std::chrono::system_clock::now();
                      *scanCloud = pclCloud;
                      new_scan = false;
                    }
                  } });

  double maxError = 0.0;
  int scansProcessed = 0;
  const int warmupScans = 2; // ignore first few scans for pass/fail

  while (!viewer->wasStopped())
  {
    while (new_scan)
    {
      std::this_thread::sleep_for(0.1s);
      world.Tick(1s);
    }

    if (refresh_view)
    {
      viewer->setCameraPosition(pose.position.x, pose.position.y, 60,
                                pose.position.x + 1, pose.position.y + 1, 0,
                                0, 0, 1);
      refresh_view = false;
    }

    // Ground-truth (relative to poseRef)
    Pose truePose = Pose(
                      Point(vehicle->GetTransform().location.x,
                            vehicle->GetTransform().location.y,
                            vehicle->GetTransform().location.z),
                      Rotate(vehicle->GetTransform().rotation.yaw * pi / 180,
                             vehicle->GetTransform().rotation.pitch * pi / 180,
                             vehicle->GetTransform().rotation.roll * pi / 180))
                    - poseRef;

    // Visualize steering ray based on current control
    viewer->removeShape("box0");
    viewer->removeShape("boxFill0");
    drawCar(truePose, 0, Color(1, 0, 0), 0.7, viewer);
    double theta = truePose.rotation.yaw;
    double stheta = control.steer * pi / 4 + theta;
    viewer->removeShape("steer");
    renderRay(viewer,
              Point(truePose.position.x + 2 * cos(theta), truePose.position.y + 2 * sin(theta), truePose.position.z),
              Point(truePose.position.x + 4 * cos(stheta), truePose.position.y + 4 * sin(stheta), truePose.position.z),
              "steer", Color(0, 1, 0));

    // Handle key inputs
    ControlState accuate(0, 0, 1);
    if (!cs.empty())
    {
      accuate = cs.back();
      cs.clear();

      // Release brake if throttle/steer inputs start coming in
      if (control.brake > 0.0f && (accuate.t != 0.0 || fabs(accuate.s) > 1e-6))
      {
        control.brake = 0.0f;
      }

      Accuate(accuate, control);
      vehicle->ApplyControl(control);
    }

    viewer->spinOnce();

    if (!new_scan)
    {
      new_scan = true;

      // Filter scan for stability
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(scanCloud);
      double filterRes = 1.0; // try 1.5-2.0 if you need more robustness
      vg.setLeafSize(filterRes, filterRes, filterRes);
      vg.filter(*cloudFiltered);

      // Pose alignment
#if USE_NDT == 1
      Eigen::Matrix4d transform_matrix = NDT(mapCloud, cloudFiltered, pose, 100);
#else
      Eigen::Matrix4d transform_matrix = ICP(mapCloud, cloudFiltered, pose, 30);
#endif
      pose = getPose(transform_matrix);

      // Establish estimated reference the first time (to compare in the same relative frame as truePose)
      if (first_est)
      {
        estRef = pose;
        first_est = false;
      }
      Pose estRel = pose - estRef;

      // Render corrected scan in world (already transformed by transform_matrix)
      PointCloudT::Ptr corrected_scan(new PointCloudT);
      pcl::transformPointCloud(*cloudFiltered, *corrected_scan, transform_matrix);
      viewer->removePointCloud("scan");
      renderPointCloud(viewer, corrected_scan, "scan", Color(1, 0, 0));

      // Draw estimated car in relative frame to match truePose drawing
      viewer->removeAllShapes();
      drawCar(truePose, 0, Color(1, 0, 0), 0.7, viewer); // red: ground truth (relative)
      drawCar(estRel, 1, Color(0, 1, 0), 0.35, viewer);  // green: estimate (relative to its own ref)

      // Error in the same (relative) frame
      double poseError = hypot(truePose.position.x - estRel.position.x,
                               truePose.position.y - estRel.position.y);

      ++scansProcessed;
      if (scansProcessed > warmupScans)
      {
        if (poseError > maxError) maxError = poseError;
      }

      double distDriven = hypot(truePose.position.x, truePose.position.y);

      // HUD
      viewer->removeShape("maxE");
      viewer->addText("Max Error: " + to_string(maxError) + " m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE", 0);
      viewer->removeShape("derror");
      viewer->addText("Pose error: " + to_string(poseError) + " m", 200, 150, 32, 1.0, 1.0, 1.0, "derror", 0);
      viewer->removeShape("dist");
      viewer->addText("Distance: " + to_string(distDriven) + " m", 200, 200, 32, 1.0, 1.0, 1.0, "dist", 0);

      // Evaluation (ignore warm-up)
      if (scansProcessed > warmupScans && (maxError > 1.2 || distDriven >= 170.0))
      {
        viewer->removeShape("eval");
        if (maxError > 1.2)
        {
          viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval", 0);
        }
        else
        {
          viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval", 0);
        }
      }

      pclCloud.points.clear();
    }
  }
  return 0;
}
