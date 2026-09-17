// Transport adapter only. Frontier search, visibility and MRTSP/DP are upstream.
// This executable has no action client and no velocity/map publisher.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <frontier_exploration_ros2/frontier_search.hpp>
#include <frontier_exploration_ros2/mrtsp_ordering.hpp>
#include <frontier_exploration_ros2/mrtsp_solver.hpp>

namespace fe = frontier_exploration_ros2;
class CandidateBridge : public rclcpp::Node {
  using Grid = nav_msgs::msg::OccupancyGrid;
  Grid::ConstSharedPtr map_, cost_;
  rclcpp::Subscription<Grid>::SharedPtr map_sub_, cost_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  double min_length_, sensor_range_;
  double age(const builtin_interfaces::msg::Time & stamp) {
    return (now() - rclcpp::Time(stamp)).seconds();
  }
  static void validate_grid(const Grid & g) {
    const auto & q = g.info.origin.orientation;
    if (g.header.frame_id != "map" || g.info.resolution <= 0 ||
        g.data.size() != size_t(g.info.width)*g.info.height ||
        std::abs(q.x)+std::abs(q.y)+std::abs(q.z) > 1e-7 ||
        std::abs(std::abs(q.w)-1) > 1e-7)
      throw std::runtime_error("invalid_or_rotated_map");
  }
public:
  CandidateBridge() : Node("zeng306_frontier_candidates"), buffer_(get_clock()), listener_(buffer_) {
    min_length_ = declare_parameter("minimum_frontier_length_m", 0.45);
    sensor_range_ = declare_parameter("sensor_effective_range_m", 3.0);
    auto qos = rclcpp::QoS(1).transient_local().reliable();
    map_sub_ = create_subscription<Grid>("/map", qos, [this](Grid::ConstSharedPtr m){map_=m;});
    cost_sub_ = create_subscription<Grid>("/global_costmap/costmap", qos, [this](Grid::ConstSharedPtr m){cost_=m;});
    service_ = create_service<std_srvs::srv::Trigger>("/zeng306/frontier_candidates",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        try { response->message = snapshot(); response->success = true; }
        catch(const std::exception & e) { response->success = false; response->message = e.what(); }
      });
  }
  std::string snapshot() {
    if (!map_ || !cost_) throw std::runtime_error("map_or_native_costmap_missing");
    if (count_publishers("/map") != 1 || count_publishers("/global_costmap/costmap") != 1)
      throw std::runtime_error("map_or_costmap_owner_not_unique");
    validate_grid(*map_); validate_grid(*cost_);
    if (age(map_->header.stamp) < -0.3 || age(map_->header.stamp) > 6.0 ||
        age(cost_->header.stamp) < -0.3 || age(cost_->header.stamp) > 3.0)
      throw std::runtime_error("stale_map_or_costmap");
    auto t = buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
    if (age(t.header.stamp) < -0.3 || age(t.header.stamp) > 0.65)
      throw std::runtime_error("stale_robot_pose");
    geometry_msgs::msg::Pose pose;
    pose.position.x=t.transform.translation.x; pose.position.y=t.transform.translation.y;
    pose.orientation=t.transform.rotation;
    auto & q = pose.orientation;
    double yaw=std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));
    fe::OccupancyGrid2d map(map_), cost(cost_);
    fe::FrontierSearchOptions opts;
    opts.min_frontier_size_cells=std::max(5, int(std::ceil(min_length_/map_->info.resolution)));
    opts.candidate_min_goal_distance_m=0.65;
    // Preserve native occupancy. Decision-map smoothing is deliberately disabled.
    auto found=fe::get_frontier(pose,map,cost,std::nullopt,0.65,true,opts);
    fe::RobotState robot{{pose.position.x,pose.position.y},yaw};
    fe::CostWeights weights{1.0,2.0};
    auto pool=fe::prune_mrtsp_candidates(found.frontiers,robot,weights,sensor_range_,0.10,0.09,{12,6});
    std::vector<fe::FrontierCandidate> candidates;
    for(auto & p:pool) candidates.push_back(p.candidate);
    auto matrix=fe::build_cost_matrix(candidates,robot,weights,sensor_range_,0.10,0.09);
    auto order=fe::solve_bounded_horizon_mrtsp_order(matrix,6);
    if(order.empty()) order=fe::greedy_mrtsp_order(matrix);
    std::ostringstream out; out<<std::setprecision(12);
    out<<"{\"upstream_commit\":\"ec530d2a813739cd25dd0c438d2365c510b9fad8\","
       <<"\"stamp\":"<<now().seconds()<<",\"map_stamp\":"<<rclcpp::Time(map_->header.stamp).seconds()
       <<",\"native_frontier_count\":"<<found.frontiers.size()<<",\"pose\":["
       <<pose.position.x<<","<<pose.position.y<<","<<yaw<<"],\"candidates\":[";
    bool comma=false;
    for(size_t i=0;i<candidates.size();++i) {
      auto & c=candidates[i]; if(!c.goal_point) continue;
      auto [x,y]=*c.goal_point;
      if(std::hypot(x-pose.position.x,y-pose.position.y)<0.65 || !std::isfinite(pool[i].score)) continue;
      geometry_msgs::msg::Pose sensor=pose; sensor.position.x=x; sensor.position.y=y;
      double heading=std::atan2(c.centroid.second-y,c.centroid.first-x);
      sensor.orientation.x=0; sensor.orientation.y=0;
      sensor.orientation.z=std::sin(heading/2); sensor.orientation.w=std::cos(heading/2);
      auto visible=fe::compute_visible_reveal_gain(sensor,map,cost,std::nullopt,sensor_range_,360,2,c.visible_reveal_bounds);
      auto rank=std::find(order.begin(),order.end(),i);
      if(comma) out<<",";
      comma=true;
      out<<"{\"kind\":\"frontier\",\"xy\":["<<x<<","<<y<<"],\"yaw\":"<<heading
         <<",\"cluster_cells\":"<<c.size<<",\"mrtsp_cost\":"<<pool[i].score
         <<",\"dp_rank\":"<<(rank==order.end()?order.size()+1:size_t(rank-order.begin())+1)
         <<",\"visible_frontier_m\":"<<(visible?visible->visible_reveal_length_m:0.0)<<"}";
    }
    out<<"]}"; return out.str();
  }
};
int main(int argc,char ** argv) {
  rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<CandidateBridge>()); rclcpp::shutdown();
}
