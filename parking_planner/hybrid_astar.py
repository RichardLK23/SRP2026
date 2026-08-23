"""
模块2：改进混合A*路径搜索
"""
import math
import heapq
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from shapely.geometry import Polygon

from config import Config
from vehicle_model import VehicleModel, generate_reeds_shepp_path
from collision_checker import CollisionChecker
from geometry import distance, angle_diff, normalize_angle


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
        self.direction = 1
        self.step_index = 0
        
    def __lt__(self, other):
        return self.f < other.f
    
    def get_pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)
    
    def get_grid_key(self, xy_res: float, theta_res: float) -> Tuple[int, int, int]:
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
        
        self.xy_res = config.hybrid_astar.XY_RESOLUTION
        self.theta_res = config.hybrid_astar.THETA_RESOLUTION
        self.max_iterations = config.hybrid_astar.MAX_ITERATIONS
        self.max_expansion = config.hybrid_astar.MAX_EXPANSION_NODES
        
        self.weight_path = config.cost.WEIGHT_PATH_LENGTH
        self.weight_gear = config.cost.WEIGHT_GEAR_CHANGE
        self.weight_steering = config.cost.WEIGHT_STEERING_CHANGE
        self.weight_reverse = config.cost.WEIGHT_REVERSE
        self.weight_guide_dist = config.cost.WEIGHT_GUIDE_DIST
        self.weight_heading = config.cost.WEIGHT_HEADING_DIFF
        
        self.motion_primitives = self._generate_primitives()
        
        self.guide_points = []
        self.guide_point_index = 0
        self.guide_reach_radius = config.hybrid_astar.GUIDE_POINT_REACH_RADIUS
        self.angle_threshold = math.radians(config.hybrid_astar.GUIDE_POINT_ANGLE_THRESHOLD)
    
    def plan(self, 
             start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             guide_points: List[Tuple[float, float, float]]) -> Optional[List[Tuple[float, float, float]]]:
        """执行混合A*搜索"""
        self.guide_points = guide_points
        self.guide_point_index = 0
        
        print(f"开始搜索: 起点={start}, 目标={goal}")
        print(f"引导点: {guide_points}")
        
        if self.collision_checker.check_vehicle_collision(start[0], start[1], start[2]):
            print("错误：起点发生碰撞")
            return None
        
        start_node = Node(start[0], start[1], start[2], 0.0)
        start_node.h = self._calculate_heuristic(start_node, goal, guide_points[-1] if guide_points else goal)
        start_node.f = start_node.h
        
        open_list = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()
        node_map: Dict[Tuple[int, int, int], Node] = {}
        node_map[start_node.get_grid_key(self.xy_res, self.theta_res)] = start_node
        
        start_time = time.time()
        iterations = 0
        
        while open_list and iterations < self.max_iterations:
            iterations += 1
            
            if time.time() - start_time > self.config.hybrid_astar.TIME_LIMIT:
                print(f"搜索超时 ({self.config.hybrid_astar.TIME_LIMIT}秒)")
                break
            
            current = heapq.heappop(open_list)
            current_key = current.get_grid_key(self.xy_res, self.theta_res)
            
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # 每1000次迭代打印进度
            if iterations % 1000 == 0:
                print(f"迭代 {iterations}: 当前位置 ({current.x:.2f}, {current.y:.2f})")
            
            # 检查是否到达目标
            if self._is_goal_reached(current, goal):
                print(f"找到路径！扩展节点数: {len(closed_set)}")
                return self._reconstruct_path(current)
            
            # 尝试RS曲线直接连接
            # 至少扩展100个节点后再尝试RS曲线，确保搜索充分
            if len(closed_set) > 100:
                if self._try_rs_connection(current, goal):
                    print(f"RS曲线连接成功！扩展节点数: {len(closed_set)}")
                    return self._reconstruct_path(current)
            
            # 更新引导点索引
            self._update_guide_point_index(current)
            
            # 生成子节点
            children = self._expand_node(current, goal)
            
            for child in children:
                child_key = child.get_grid_key(self.xy_res, self.theta_res)
                
                if child_key in closed_set:
                    continue
                
                if self.collision_checker.check_vehicle_collision(child.x, child.y, child.theta):
                    continue
                
                child.h = self._calculate_heuristic(child, goal, self._get_current_guide_point())
                child.f = child.g + child.h
                
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
        
        # 前进基元 - 使用更明显的转向角度
        steering_angles = [0, 0.5*max_steering, max_steering]
        for steering in steering_angles:
            primitives.append((steering, 1, step_size))
            if steering != 0:
                primitives.append((-steering, 1, step_size))
        
        # 后退基元
        reverse_steering = [0, 0.3*max_steering, 0.6*max_steering]
        for steering in reverse_steering:
            primitives.append((steering, -1, step_size * 0.8))
            if steering != 0:
                primitives.append((-steering, -1, step_size * 0.8))
        
        return primitives
    
    def _expand_node(self, node: Node, goal: Tuple[float, float, float]) -> List[Node]:
        """扩展节点"""
        children = []
        use_reverse = self._should_use_reverse(node)
        
        for steering, direction, step_size in self.motion_primitives:
            if not use_reverse and direction < 0:
                continue
            
            # 应用运动基元
            x_new, y_new, theta_new = self.vehicle.apply_motion_primitive(
                node.x, node.y, node.theta, steering, direction, step_size
            )
            
            # 碰撞检测 - 提前过滤
            if self.collision_checker.check_vehicle_collision(x_new, y_new, theta_new):
                continue
            
            # 检查线段碰撞
            if self.collision_checker.check_line_collision(
                (node.x, node.y), (x_new, y_new)
            ):
                continue
            
            # 计算代价值
            g_new = node.g + step_size
            
            if direction != node.direction:
                g_new += self.weight_gear * 0.5
            
            steering_change = abs(steering - node.steering)
            g_new += self.weight_steering * steering_change
            
            if direction < 0:
                g_new += self.weight_reverse * step_size
            
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
        
        dx = current_guide[0] - node.x
        dy = current_guide[1] - node.y
        if dx == 0 and dy == 0:
            return True
        
        target_heading = math.atan2(dy, dx)
        heading_diff = abs(angle_diff(node.theta, target_heading))
        
        return heading_diff > self.angle_threshold
    
    def _calculate_heuristic(self, node: Node, goal: Tuple[float, float, float],
                            guide_point: Optional[Tuple[float, float, float]]) -> float:
        """计算启发值"""
        h = 0.0
        
        if guide_point:
            dist_to_guide = distance((node.x, node.y), (guide_point[0], guide_point[1]))
            heading_diff = abs(angle_diff(node.theta, guide_point[2]))
            
            # ===== 增加引导点权重，让搜索更倾向于跟随引导点 =====
            h += self.weight_guide_dist * dist_to_guide * 2.0  # 增加权重
            h += self.weight_heading * heading_diff * 1.5
        
        dist_to_goal = distance((node.x, node.y), (goal[0], goal[1]))
        h += dist_to_goal
        
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
    
    def _try_rs_connection(self, node: Node, goal: Tuple[float, float, float]) -> bool:
        """尝试用RS曲线连接到目标 - 只在距离足够近且无碰撞时使用"""
        node_pose = (node.x, node.y, node.theta)
        goal_pose = (goal[0], goal[1], goal[2])
        
        # 检查距离 - 只有5米内才使用RS曲线
        dist = distance((node.x, node.y), (goal[0], goal[1]))
        if dist > 5.0:
            return False
        
        # 检查航向是否大致朝向目标 - 航向差超过30度不使用RS曲线
        angle_to_goal = math.atan2(goal[1] - node.y, goal[0] - node.x)
        heading_diff = abs(angle_diff(node.theta, angle_to_goal))
        if heading_diff > math.radians(30):
            return False
        
        # 生成RS路径 (实际上是直线)
        rs_path = generate_reeds_shepp_path(node_pose, goal_pose,
                                            self.vehicle.turning_radius,
                                            self.config.vehicle.STEP_SIZE)
        
        if rs_path and len(rs_path) > 2:
            # ===== 关键修复：检查RS路径上的所有点是否碰撞 =====
            for i, (x, y, theta) in enumerate(rs_path):
                if self.collision_checker.check_vehicle_collision(x, y, theta):
                    print(f"  RS路径点 {i} 碰撞: ({x:.2f}, {y:.2f})，放弃RS连接")
                    return False
            
            # 检查路径线段
            for i in range(len(rs_path) - 1):
                p1 = (rs_path[i][0], rs_path[i][1])
                p2 = (rs_path[i+1][0], rs_path[i+1][1])
                if self.collision_checker.check_line_collision(p1, p2):
                    print(f"  RS线段 {i}-{i+1} 碰撞，放弃RS连接")
                    return False
            
            print(f"  RS曲线安全！距离目标 {dist:.2f}m")
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
        
        # 如果路径点太少，进行插值
        if len(path) < 5:
            print(f"路径点太少 ({len(path)}个)，进行插值")
            path = self._interpolate_path(path, 50)
        
        # 计算路径总长度
        total_len = 0
        for i in range(len(path) - 1):
            total_len += distance((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]))
        print(f"路径总长度: {total_len:.2f}m, 路径点数: {len(path)}")
        
        return path
    
    def _interpolate_path(self, path: List[Tuple[float, float, float]], 
                          num_points: int) -> List[Tuple[float, float, float]]:
        """插值路径点"""
        if len(path) < 2:
            return path
        
        # 计算总长度
        segments = []
        total_length = 0
        for i in range(len(path) - 1):
            dist = distance((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]))
            segments.append(dist)
            total_length += dist
        
        if total_length < 0.001:
            # 如果总长度为0，生成从起点到终点的直线
            print("路径长度为0，生成直线路径")
            return self._generate_straight_path(path, num_points)
        
        interpolated = []
        for i in range(num_points):
            t = i / (num_points - 1)
            target_dist = t * total_length
            
            cum_dist = 0
            for j, seg_len in enumerate(segments):
                if cum_dist + seg_len >= target_dist or j == len(segments) - 1:
                    local_t = (target_dist - cum_dist) / seg_len if seg_len > 0.001 else 0
                    local_t = max(0, min(1, local_t))
                    
                    x = path[j][0] + local_t * (path[j+1][0] - path[j][0])
                    y = path[j][1] + local_t * (path[j+1][1] - path[j][1])
                    theta = normalize_angle(path[j][2] + local_t * angle_diff(path[j][2], path[j+1][2]))
                    
                    interpolated.append((x, y, theta))
                    break
                
                cum_dist += seg_len
        
        # 确保包含最后一个点
        if len(interpolated) > 0:
            interpolated[-1] = (path[-1][0], path[-1][1], path[-1][2])
        
        return interpolated
    
    def _generate_straight_path(self, path: List[Tuple[float, float, float]], 
                                num_points: int) -> List[Tuple[float, float, float]]:
        """生成直线路径"""
        if len(path) < 2:
            return [(path[0][0], path[0][1], path[0][2]) for _ in range(num_points)]
        
        start = path[0]
        end = path[-1]
        
        straight_path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            theta = normalize_angle(start[2] + t * angle_diff(start[2], end[2]))
            straight_path.append((x, y, theta))
        
        return straight_path