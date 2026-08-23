"""
模块1：简化可视图构建与引导点提取
"""
import math
import heapq
from typing import List, Tuple, Optional, Set
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from config import Config
from collision_checker import CollisionChecker
from geometry import distance, compute_heading, normalize_angle


class SVGGuidePointExtractor:
    """简化可视图引导点提取器"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        # 确保障碍物是有效的 Polygon 对象
        self.obstacle_polys = self._validate_obstacles(obstacle_polys)
        self.collision_checker = CollisionChecker(self.obstacle_polys, config)
        self.view_margin = config.svg.VIEW_WINDOW_MARGIN
        self.simplify_tolerance = config.svg.SIMPLIFY_TOLERANCE
        self.use_convex_hull = config.svg.USE_CONVEX_HULL
        self.min_edge_length = config.svg.MIN_EDGE_LENGTH
    
    def _validate_obstacles(self, obstacle_polys: List) -> List[Polygon]:
        """验证并转换障碍物为有效的 Polygon 对象"""
        validated = []
        for poly in obstacle_polys:
            if poly is None:
                continue
            if isinstance(poly, Polygon):
                if not poly.is_empty and poly.is_valid:
                    validated.append(poly)
                else:
                    # 尝试修复无效多边形
                    try:
                        fixed = poly.buffer(0)
                        if fixed.is_valid:
                            validated.append(fixed)
                    except:
                        pass
            elif hasattr(poly, '__iter__') and len(poly) >= 3:
                # 坐标列表转换为 Polygon
                try:
                    p = Polygon(poly)
                    if p.is_valid and not p.is_empty:
                        validated.append(p)
                except:
                    pass
        return validated
    
    def extract_guide_points(self, 
                            start: Tuple[float, float, float],
                            goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """提取引导点序列"""
        # 1. 简化障碍物
        simplified_obstacles = self._simplify_obstacles()
        
        # 2. 提取节点
        nodes = self._extract_nodes(start, goal, simplified_obstacles)
        
        # 如果节点太少，返回直接路径
        if len(nodes) < 2:
            print("警告：节点太少，使用直接路径")
            return [start, goal]
        
        # 3. 构建可视图
        edges = self._build_visibility_graph(nodes, simplified_obstacles)
        
        # 4. Dijkstra搜索
        path_nodes = self._dijkstra_search(nodes, edges, start, goal)
        
        if not path_nodes or len(path_nodes) < 2:
            print("警告：未找到引导路径，使用直线路径")
            return [start, goal]
        
        # 5. 补全航向角
        guide_points = self._assign_headings(path_nodes, start[2], goal[2])
        
        return guide_points
    
    def _simplify_obstacles(self) -> List[Polygon]:
        """简化障碍物多边形"""
        simplified = []
        for poly in self.obstacle_polys:
            try:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                
                if self.use_convex_hull:
                    poly = poly.convex_hull
                
                if self.simplify_tolerance > 0:
                    poly = poly.simplify(self.simplify_tolerance, preserve_topology=True)
                
                if poly.is_valid and not poly.is_empty:
                    simplified.append(poly)
            except Exception as e:
                print(f"简化障碍物警告: {e}")
                continue
        
        return simplified
    
    def _extract_nodes(self, 
                      start: Tuple[float, float, float],
                      goal: Tuple[float, float, float],
                      obstacles: List[Polygon]) -> List[Tuple[float, float]]:
        """提取可视图节点"""
        nodes = []
        
        # 添加起点和终点
        nodes.append((float(start[0]), float(start[1])))
        nodes.append((float(goal[0]), float(goal[1])))
        
        # 计算可视窗口边界
        min_x = min(start[0], goal[0]) - self.view_margin
        max_x = max(start[0], goal[0]) + self.view_margin
        min_y = min(start[1], goal[1]) - self.view_margin
        max_y = max(start[1], goal[1]) + self.view_margin
        
        # 提取障碍物顶点
        for poly in obstacles:
            try:
                # 检查多边形是否在可视窗口内
                if not self._is_polygon_in_window(poly, min_x, max_x, min_y, max_y):
                    continue
                
                # 获取顶点
                coords = list(poly.exterior.coords)[:-1]  # 移除重复的最后一个点
                for coord in coords:
                    if min_x <= coord[0] <= max_x and min_y <= coord[1] <= max_y:
                        nodes.append((float(coord[0]), float(coord[1])))
            except Exception as e:
                print(f"提取节点警告: {e}")
                continue
        
        # 移除重复节点
        unique_nodes = self._remove_duplicate_nodes(nodes)
        
        return unique_nodes
    
    def _is_polygon_in_window(self, poly: Polygon, 
                              min_x: float, max_x: float,
                              min_y: float, max_y: float) -> bool:
        """检查多边形是否在可视窗口内"""
        try:
            bounds = poly.bounds
            if bounds[2] < min_x or bounds[0] > max_x or bounds[3] < min_y or bounds[1] > max_y:
                return False
            return True
        except:
            return False
    
    def _remove_duplicate_nodes(self, nodes: List[Tuple[float, float]], 
                               tolerance: float = 0.01) -> List[Tuple[float, float]]:
        """移除重复节点"""
        if not nodes:
            return []
        
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
        if len(nodes) < 2:
            return []
        
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
        if len(nodes) < 2:
            return None
        
        n = len(nodes)
        start_idx = 0
        goal_idx = 1
        
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
        
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == goal_idx:
                break
            
            for v, weight in adj[u]:
                if v in visited:
                    continue
                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        # 重建路径
        if dist[goal_idx] == float('inf') or prev[goal_idx] == -1:
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
                heading = start_heading
            elif i == len(path_nodes) - 1:
                heading = goal_heading
            else:
                prev_node = path_nodes[i-1]
                next_node = path_nodes[i+1]
                dir1 = compute_heading(prev_node, node)
                dir2 = compute_heading(node, next_node)
                angle_diff_val = normalize_angle(dir2 - dir1)
                heading = normalize_angle(dir1 + 0.5 * angle_diff_val)
            
            guide_points.append((node[0], node[1], heading))
        
        return guide_points