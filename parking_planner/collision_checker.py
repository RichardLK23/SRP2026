"""
碰撞检测模块
"""
from typing import List, Tuple
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.strtree import STRtree
from geometry import create_vehicle_rect
import math


class CollisionChecker:
    """碰撞检测器"""
    
    def __init__(self, obstacle_polys: List[Polygon], config):
        self.config = config
        # 确保所有障碍物都是 Polygon 类型
        self.obstacle_polys = []
        for p in obstacle_polys:
            if p is not None:
                if isinstance(p, Polygon) and p.is_valid and not p.is_empty:
                    self.obstacle_polys.append(p)
                elif isinstance(p, MultiPolygon):
                    # 展开 MultiPolygon
                    for geom in p.geoms:
                        if geom.is_valid and not geom.is_empty:
                            self.obstacle_polys.append(geom)
        
        # 构建空间索引
        self.spatial_index = STRtree(self.obstacle_polys) if self.obstacle_polys else None
        self.vehicle_length = config.vehicle.LENGTH
        self.vehicle_width = config.vehicle.WIDTH
        self.clearance = config.cost.OBSTACLE_CLEARANCE
    
    def check_line_collision(self, p1: Tuple[float, float], 
                             p2: Tuple[float, float]) -> bool:
        """检查线段是否与障碍物碰撞"""
        if not self.obstacle_polys:
            return False
        
        # 确保坐标是float
        try:
            p1 = (float(p1[0]), float(p1[1]))
            p2 = (float(p2[0]), float(p2[1]))
        except:
            return False
        
        line = LineString([p1, p2])
        
        try:
            if self.spatial_index:
                # 查询可能与线段相交的障碍物
                candidates = self.spatial_index.query(line)
                for poly in candidates:
                    if poly is None:
                        continue
                    if line.intersects(poly):
                        # 如果交点只是端点，不算碰撞
                        intersection = line.intersection(poly)
                        if intersection.is_empty:
                            continue
                        if intersection.geom_type == 'Point':
                            coords = list(intersection.coords)
                            if len(coords) == 1:
                                pt = coords[0]
                                if self._is_endpoint(pt, p1, p2):
                                    continue
                        return True
            else:
                for poly in self.obstacle_polys:
                    if poly is not None and line.intersects(poly):
                        return True
        except Exception as e:
            # 静默处理，避免大量警告
            pass
        
        return False
    
    def _is_endpoint(self, pt, p1, p2, tolerance=0.001):
        """检查点是否在线段端点"""
        dist_to_p1 = math.sqrt((pt[0] - p1[0])**2 + (pt[1] - p1[1])**2)
        dist_to_p2 = math.sqrt((pt[0] - p2[0])**2 + (pt[1] - p2[1])**2)
        return dist_to_p1 < tolerance or dist_to_p2 < tolerance
    
    def check_vehicle_collision(self, x: float, y: float, theta: float) -> bool:
        """检查车辆在当前位置是否碰撞"""
        if not self.obstacle_polys:
            return False
        
        try:
            vehicle_poly = create_vehicle_rect(x, y, theta, 
                                              self.vehicle_length, 
                                              self.vehicle_width)
            
            if self.clearance > 0:
                vehicle_poly = vehicle_poly.buffer(self.clearance)
            
            if self.spatial_index:
                candidates = self.spatial_index.query(vehicle_poly)
                for poly in candidates:
                    if poly is not None and vehicle_poly.intersects(poly):
                        return True
            else:
                for poly in self.obstacle_polys:
                    if poly is not None and vehicle_poly.intersects(poly):
                        return True
        except:
            pass
        
        return False
    
    def check_path_collision(self, path: List[Tuple[float, float, float]]) -> bool:
        """检查路径上所有点是否碰撞"""
        if not path:
            return False
        
        # 检查路径点
        for x, y, theta in path:
            if self.check_vehicle_collision(x, y, theta):
                return True
        
        return False
    
    def get_clearance_to_obstacles(self, x: float, y: float, theta: float) -> float:
        """获取车辆到最近障碍物的距离"""
        if not self.obstacle_polys:
            return 10.0  # 无障碍物时返回较大值
        
        try:
            vehicle_poly = create_vehicle_rect(x, y, theta,
                                              self.vehicle_length,
                                              self.vehicle_width)
            
            min_dist = float('inf')
            for poly in self.obstacle_polys:
                if poly is not None:
                    dist = vehicle_poly.distance(poly)
                    if dist < min_dist:
                        min_dist = dist
            return min_dist if min_dist != float('inf') else 10.0
        except:
            return 10.0