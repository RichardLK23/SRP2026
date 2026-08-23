"""
模块1：简化可视图构建与引导点提取
"""
import math
import heapq
from typing import List, Tuple, Optional, Set
from shapely.geometry import Polygon, Point, LineString

from ..config import Config
from .collision_checker import CollisionChecker
from ...utils.geometry import distance, compute_heading, normalize_angle


class SVGGuidePointExtractor:
    """简化可视图引导点提取器"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        self.obstacle_polys = obstacle_polys
        self.collision_checker = CollisionChecker(obstacle_polys, config)
        
        # 参数
        self.view_margin = config.svg.VIEW_WINDOW_MARGIN
        self.simplify_tolerance = config.svg.SIMPLIFY_TOLERANCE
        self.use_convex_hull = config.svg.USE_CONVEX_HULL
        self.min_edge_length = config.svg.MIN_EDGE_LENGTH
    
    def extract_guide_points(self, 
                            start: Tuple[float, float, float],
                            goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        提取引导点序列
        
        Args:
            start: 起始位姿 (x, y, theta)
            goal: 目标位姿 (x, y, theta)
        
        Returns:
            引导点序列，包含起点和终点，格式 [(x, y, theta), ...]
        """
        # 1. 预处理障碍物
        simplified_obstacles = self._simplify_obstacles()
        
        # 2. 构建可视窗口
        nodes = self._extract_nodes(start, goal, simplified_obstacles)
        
        # 3. 构建可视图
        edges = self._build_visibility_graph(nodes, simplified_obstacles)
        
        # 4. Dijkstra搜索最短路径
        path_nodes = self._dijkstra_search(nodes, edges, start, goal)
        
        if not path_nodes:
            # 如果没有找到路径，返回简单的直线路径
            print("警告：未找到引导路径，使用直线路径")
            return [start, goal]
        
        # 5. 补全航向角
        guide_points = self._assign_headings(path_nodes, start[2], goal[2])
        
        return guide_points
    
    def _simplify_obstacles(self) -> List[Polygon]:
        """简化障碍物多边形"""
        simplified = []
        for poly in self.obstacle_polys:
            if self.use_convex_hull:
                poly = poly.convex_hull
            if self.simplify_tolerance > 0:
                poly = poly.simplify(self.simplify_tolerance, preserve_topology=True)
            simplified.append(poly)
        return simplified
    
    def _extract_nodes(self, 
                      start: Tuple[float, float, float],
                      goal: Tuple[float, float, float],
                      obstacles: List[Polygon]) -> List[Tuple[float, float]]:
        """提取可视图节点"""
        nodes = []
        
        # 添加起点和终点（只取x,y坐标）
        nodes.append((start[0], start[1]))
        nodes.append((goal[0], goal[1]))
        
        # 计算可视窗口边界
        min_x = min(start[0], goal[0]) - self.view_margin
        max_x = max(start[0], goal[0]) + self.view_margin
        min_y = min(start[1], goal[1]) - self.view_margin
        max_y = max(start[1], goal[1]) + self.view_margin
        
        # 提取障碍物顶点
        for poly in obstacles:
            # 检查多边形是否在可视窗口内
            if not self._is_polygon_in_window(poly, min_x, max_x, min_y, max_y):
                continue
            
            # 获取顶点
            coords = list(poly.exterior.coords)[:-1]  # 移除重复的最后一个点
            for coord in coords:
                # 只添加在可视窗口内的点
                if min_x <= coord[0] <= max_x and min_y <= coord[1] <= max_y:
                    nodes.append(coord)
        
        # 移除重复节点
        unique_nodes = self._remove_duplicate_nodes(nodes)
        
        return unique_nodes
    
    def _is_polygon_in_window(self, poly: Polygon, 
                              min_x: float, max_x: float,
                              min_y: float, max_y: float) -> bool:
        """检查多边形是否在可视窗口内"""
        bounds = poly.bounds
        if bounds[2] < min_x or bounds[0] > max_x or bounds[3] < min_y or bounds[1] > max_y:
            return False
        return True
    
    def _remove_duplicate_nodes(self, nodes: List[Tuple[float, float]], 
                               tolerance: float = 0.01) -> List[Tuple[float, float]]:
        """移除重复节点"""
        unique = []
        for node in nodes:
            is_duplicate = False
            for existing in unique:
                if distance(node, existing) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(node)
        return unique
    
    def _build_visibility_graph(self, 
                               nodes: List[Tuple[float, float]],
                               obstacles: List[Polygon]) -> List[Tuple[int, int, float]]:
        """构建可视图边"""
        edges = []
        n = len(nodes)
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = nodes[i]
                p2 = nodes[j]
                dist = distance(p1, p2)
                
                # 跳过太短的边
                if dist < self.min_edge_length:
                    continue
                
                # 检查是否与障碍物碰撞
                if not self.collision_checker.check_line_collision(p1, p2):
                    edges.append((i, j, dist))
        
        return edges
    
    def _dijkstra_search(self, 
                        nodes: List[Tuple[float, float]],
                        edges: List[Tuple[int, int, float]],
                        start: Tuple[float, float, float],
                        goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float]]]:
        """Dijkstra搜索最短路径"""
        n = len(nodes)
        
        # 找到起点和终点的索引
        start_idx = 0  # 起点在第一个
        goal_idx = 1   # 终点在第二个
        
        # 构建邻接表
        adj = [[] for _ in range(n)]
        for i, j, weight in edges:
            adj[i].append((j, weight))
            adj[j].append((i, weight))
        
        # Dijkstra算法
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start_idx] = 0
        pq = [(0, start_idx)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            
            if u == goal_idx:
                break
            
            for v, weight in adj[u]:
                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        # 重建路径
        if dist[goal_idx] == float('inf'):
            return None
        
        path = []
        curr = goal_idx
        while curr != -1:
            path.append(nodes[curr])
            curr = prev[curr]
        path.reverse()
        
        return path
    
    def _assign_headings(self, 
                        path_nodes: List[Tuple[float, float]],
                        start_heading: float,
                        goal_heading: float) -> List[Tuple[float, float, float]]:
        """为路径节点分配航向角"""
        if len(path_nodes) < 2:
            return [(path_nodes[0][0], path_nodes[0][1], start_heading)]
        
        guide_points = []
        
        for i, node in enumerate(path_nodes):
            if i == 0:
                # 起点使用给定的航向
                heading = start_heading
            elif i == len(path_nodes) - 1:
                # 终点使用给定的航向
                heading = goal_heading
            else:
                # 计算平滑航向：前后方向的加权平均
                prev_node = path_nodes[i-1]
                next_node = path_nodes[i+1]
                
                # 前段方向
                dir1 = compute_heading(prev_node, node)
                # 后段方向
                dir2 = compute_heading(node, next_node)
                
                # 平滑插值
                angle_diff = normalize_angle(dir2 - dir1)
                heading = normalize_angle(dir1 + 0.5 * angle_diff)
            
            guide_points.append((node[0], node[1], heading))
        
        return guide_points