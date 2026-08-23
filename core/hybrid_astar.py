"""
模块2：改进混合A*路径搜索
"""
import math
import heapq
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from shapely.geometry import Polygon

from ..config import Config
from ..core.vehicle_model import VehicleModel, generate_reeds_shepp_path
from ..core.collision_checker import CollisionChecker
from ..utils.geometry import distance, angle_diff, normalize_angle  # 确保这些导入存在


class Node:
    """混合A*搜索节点"""
    
    def __init__(self, x: float, y: float, theta: float, 
                 g: float = float('inf'), h: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None
        self.steering = 0.0
        self.direction = 1  # 1前进，-1后退
        self.step_index = 0
        
    def __lt__(self, other):
        return self.f < other.f
    
    def get_pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)
    
    def get_grid_key(self, xy_res: float, theta_res: float) -> Tuple[int, int, int]:
        """获取网格键值用于状态去重"""
        x_idx = int(round(self.x / xy_res))
        y_idx = int(round(self.y / xy_res))
        theta_idx = int(round(math.degrees(self.theta) / theta_res))
        theta_idx = theta_idx % int(360 / theta_res)
        return (x_idx, y_idx, theta_idx)


class HybridAStarPlanner:
    """改进混合A*路径规划器"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        self.vehicle = VehicleModel(config)
        self.collision_checker = CollisionChecker(obstacle_polys, config)
        
        # 搜索参数
        self.xy_res = config.hybrid_astar.XY_RESOLUTION
        self.theta_res = config.hybrid_astar.THETA_RESOLUTION
        self.max_iterations = config.hybrid_astar.MAX_ITERATIONS
        self.max_expansion = config.hybrid_astar.MAX_EXPANSION_NODES
        
        # 代价权重
        self.weight_path = config.cost.WEIGHT_PATH_LENGTH
        self.weight_gear = config.cost.WEIGHT_GEAR_CHANGE
        self.weight_steering = config.cost.WEIGHT_STEERING_CHANGE
        self.weight_reverse = config.cost.WEIGHT_REVERSE
        self.weight_guide_dist = config.cost.WEIGHT_GUIDE_DIST
        self.weight_heading = config.cost.WEIGHT_HEADING_DIFF
        
        # 运动基元
        self.motion_primitives = self._generate_primitives()
        
        # 引导点相关
        self.guide_points = []
        self.guide_point_index = 0
        self.guide_reach_radius = config.hybrid_astar.GUIDE_POINT_REACH_RADIUS
        self.angle_threshold = math.radians(config.hybrid_astar.GUIDE_POINT_ANGLE_THRESHOLD)
    
    def plan(self, 
             start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             guide_points: List[Tuple[float, float, float]]) -> Optional[List[Tuple[float, float, float]]]:
        """
        执行混合A*搜索
        
        Args:
            start: 起始位姿 (x, y, theta)
            goal: 目标位姿 (x, y, theta)
            guide_points: 引导点序列
        
        Returns:
            粗糙路径位姿序列
        """
        self.guide_points = guide_points
        self.guide_point_index = 0
        
        # 检查起点和终点是否有效
        if self.collision_checker.check_vehicle_collision(start[0], start[1], start[2]):
            print("错误：起点发生碰撞")
            return None
        
        if self.collision_checker.check_vehicle_collision(goal[0], goal[1], goal[2]):
            print("警告：终点发生碰撞，尝试微调")
            # 这里可以尝试微调终点位置
        
        # 反向搜索：从目标向起点搜索
        start_node = Node(start[0], start[1], start[2], 0.0)
        start_node.h = self._calculate_heuristic(start_node, goal, guide_points[-1] if guide_points else goal)
        start_node.f = start_node.h
        
        # Open和Closed列表
        open_list = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()
        node_map: Dict[Tuple[int, int, int], Node] = {}
        node_map[start_node.get_grid_key(self.xy_res, self.theta_res)] = start_node
        
        start_time = time.time()
        iterations = 0
        
        while open_list and iterations < self.max_iterations:
            iterations += 1
            
            # 检查时间限制
            if time.time() - start_time > self.config.hybrid_astar.TIME_LIMIT:
                print(f"搜索超时 ({self.config.hybrid_astar.TIME_LIMIT}秒)")
                break
            
            # 弹出最优节点
            current = heapq.heappop(open_list)
            current_key = current.get_grid_key(self.xy_res, self.theta_res)
            
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # 检查是否到达目标
            if self._is_goal_reached(current, goal):
                print(f"找到路径！扩展节点数: {len(closed_set)}")
                return self._reconstruct_path(current)
            
            # 尝试RS曲线直接连接
            if self._try_rs_connection(current, start_node, goal):
                print(f"RS曲线连接成功！扩展节点数: {len(closed_set)}")
                return self._reconstruct_path(current)
            
            # 更新引导点索引
            self._update_guide_point_index(current)
            
            # 生成子节点
            children = self._expand_node(current, goal)
            
            for child in children:
                child_key = child.get_grid_key(self.xy_res, self.theta_res)
                
                # 跳过已在closed set中的节点
                if child_key in closed_set:
                    continue
                
                # 检查碰撞
                if self.collision_checker.check_vehicle_collision(child.x, child.y, child.theta):
                    continue
                
                # 计算启发值
                child.h = self._calculate_heuristic(child, goal, self._get_current_guide_point())
                child.f = child.g + child.h
                
                # 更新open list
                if child_key not in node_map or child.g < node_map[child_key].g:
                    node_map[child_key] = child
                    heapq.heappush(open_list, child)
        
        print(f"搜索失败，扩展节点数: {len(closed_set)}")
        return None
    
    def _generate_primitives(self) -> List[Tuple[float, int, float]]:
        """生成运动基元列表"""
        primitives = []
        max_steering = self.config.vehicle.max_steering_rad
        step_size = self.config.vehicle.STEP_SIZE
        
        # 前进基元
        steering_angles = [0, 0.3*max_steering, 0.6*max_steering, max_steering]
        for steering in steering_angles:
            primitives.append((steering, 1, step_size))
            if steering != 0:
                primitives.append((-steering, 1, step_size))
        
        # 后退基元（默认启用，但在某些情况下会被过滤）
        reverse_steering = [0, 0.2*max_steering, 0.4*max_steering]
        for steering in reverse_steering:
            primitives.append((steering, -1, step_size * 0.8))
            if steering != 0:
                primitives.append((-steering, -1, step_size * 0.8))
        
        return primitives
    
    def _expand_node(self, node: Node, goal: Tuple[float, float, float]) -> List[Node]:
        """扩展节点"""
        children = []
        
        # 确定是否使用后退基元
        use_reverse = self._should_use_reverse(node)
        
        for steering, direction, step_size in self.motion_primitives:
            # 如果不需要后退，过滤掉后退基元
            if not use_reverse and direction < 0:
                continue
            
            # 应用运动基元
            x_new, y_new, theta_new = self.vehicle.apply_motion_primitive(
                node.x, node.y, node.theta, steering, direction, step_size
            )
            
            # 计算代价值
            g_new = node.g + step_size
            
            # 换挡惩罚
            if direction != node.direction:
                g_new += self.weight_gear * 0.5
            
            # 转向变化惩罚
            steering_change = abs(steering - node.steering)
            g_new += self.weight_steering * steering_change
            
            # 倒车惩罚
            if direction < 0:
                g_new += self.weight_reverse * step_size
            
            # 创建子节点
            child = Node(x_new, y_new, theta_new, g_new)
            child.parent = node
            child.steering = steering
            child.direction = direction
            child.step_index = node.step_index + 1
            
            children.append(child)
        
        return children
    
    def _should_use_reverse(self, node: Node) -> bool:
        """判断是否应该使用后退基元"""
        current_guide = self._get_current_guide_point()
        if current_guide is None:
            return True
        
        # 计算当前航向与引导点方向的夹角
        dx = current_guide[0] - node.x
        dy = current_guide[1] - node.y
        if dx == 0 and dy == 0:
            return True
        
        target_heading = math.atan2(dy, dx)
        heading_diff = abs(angle_diff(node.theta, target_heading))
        
        # 如果航向偏差小于阈值，不需要倒车
        return heading_diff > self.angle_threshold
    
    def _calculate_heuristic(self, node: Node, goal: Tuple[float, float, float],
                            guide_point: Optional[Tuple[float, float, float]]) -> float:
        """计算启发值"""
        h = 0.0
        
        if guide_point:
            # 到引导点的距离和航向偏差
            dist_to_guide = distance((node.x, node.y), (guide_point[0], guide_point[1]))
            heading_diff = abs(angle_diff(node.theta, guide_point[2]))
            
            h += self.weight_guide_dist * dist_to_guide
            h += self.weight_heading * heading_diff
        
        # 到目标的距离
        dist_to_goal = distance((node.x, node.y), (goal[0], goal[1]))
        h += dist_to_goal  # 基础距离启发
        
        # 障碍物距离惩罚
        clearance = self.collision_checker.get_clearance_to_obstacles(
            node.x, node.y, node.theta
        )
        if clearance < self.config.cost.OBSTACLE_CLEARANCE * 2:
            h += self.config.cost.WEIGHT_OBSTACLE_DIST * (1.0 / (clearance + 0.01))
        
        return h
    
    def _get_current_guide_point(self) -> Optional[Tuple[float, float, float]]:
        """获取当前引导点"""
        if self.guide_point_index < len(self.guide_points):
            return self.guide_points[self.guide_point_index]
        return None
    
    def _update_guide_point_index(self, node: Node):
        """更新引导点索引"""
        for i in range(self.guide_point_index, len(self.guide_points)):
            guide = self.guide_points[i]
            dist = distance((node.x, node.y), (guide[0], guide[1]))
            if dist < self.guide_reach_radius:
                self.guide_point_index = i + 1
            else:
                break
    
    def _is_goal_reached(self, node: Node, goal: Tuple[float, float, float]) -> bool:
        """检查是否到达目标"""
        dist = distance((node.x, node.y), (goal[0], goal[1]))
        heading_diff = abs(angle_diff(node.theta, goal[2]))
        
        return (dist < self.config.cost.GOAL_REACHED_THRESHOLD and 
                heading_diff < math.radians(self.config.cost.GOAL_HEADING_THRESHOLD))
    
    def _try_rs_connection(self, node: Node, start_node: Node,
                          goal: Tuple[float, float, float]) -> bool:
        """尝试用RS曲线连接"""
        # 尝试连接到起点（反向搜索）
        start_pose = (start_node.x, start_node.y, start_node.theta)
        node_pose = (node.x, node.y, node.theta)
        
        rs_path = generate_reeds_shepp_path(node_pose, start_pose,
                                           self.vehicle.turning_radius,
                                           self.config.vehicle.STEP_SIZE)
        
        if rs_path:
            # 检查路径是否碰撞
            if not self.collision_checker.check_path_collision(rs_path):
                # 路径有效，将其连接到当前节点
                node.rs_path = rs_path
                return True
        
        return False
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float, float]]:
        """重构路径"""
        path = []
        current = node
        
        # 如果有RS路径，添加进去
        if hasattr(current, 'rs_path') and current.rs_path:
            path.extend(current.rs_path)
        
        # 反向追踪父节点
        while current.parent is not None:
            path.append(current.get_pose())
            current = current.parent
        
        path.append(current.get_pose())
        path.reverse()
        
        return path