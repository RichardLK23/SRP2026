"""
碰撞检测模块
"""
from typing import List, Tuple
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree

from ...utils.geometry import create_vehicle_rect


class CollisionChecker:
    """碰撞检测器"""
    
    def __init__(self, obstacle_polys: List[Polygon], config):
        self.obstacle_polys = obstacle_polys
        self.config = config
        # 构建空间索引
        self.spatial_index = STRtree(obstacle_polys) if obstacle_polys else None
        
        # 车辆参数
        self.vehicle_length = config.vehicle.LENGTH
        self.vehicle_width = config.vehicle.WIDTH
        self.clearance = config.cost.OBSTACLE_CLEARANCE
    
    def check_line_collision(self, p1: Tuple[float, float], 
                             p2: Tuple[float, float]) -> bool:
        """检查线段是否与障碍物碰撞"""
        if not self.obstacle_polys:
            return False
        
        line = LineString([p1, p2])
        
        # 使用空间索引加速
        if self.spatial_index:
            candidates = self.spatial_index.query(line)
            for poly in candidates:
                if line.intersects(poly):
                    # 检查是否只是端点接触
                    if line.touches(poly) and len(line.intersection(poly).coords) <= 1:
                        continue
                    return True
        else:
            for poly in self.obstacle_polys:
                if line.intersects(poly):
                    return True
        
        return False
    
    def check_vehicle_collision(self, x: float, y: float, theta: float) -> bool:
        """检查车辆在当前位置是否碰撞"""
        if not self.obstacle_polys:
            return False
        
        vehicle_poly = create_vehicle_rect(x, y, theta, 
                                          self.vehicle_length, 
                                          self.vehicle_width)
        
        # 扩展安全间隙
        if self.clearance > 0:
            vehicle_poly = vehicle_poly.buffer(self.clearance)
        
        if self.spatial_index:
            candidates = self.spatial_index.query(vehicle_poly)
            for poly in candidates:
                if vehicle_poly.intersects(poly):
                    return True
        else:
            for poly in self.obstacle_polys:
                if vehicle_poly.intersects(poly):
                    return True
        
        return False
    
    def check_path_collision(self, path: List[Tuple[float, float, float]]) -> bool:
        """检查路径上所有点是否碰撞"""
        for x, y, theta in path:
            if self.check_vehicle_collision(x, y, theta):
                return True
        return False
    
    def get_clearance_to_obstacles(self, x: float, y: float, theta: float) -> float:
        """获取车辆到最近障碍物的距离"""
        if not self.obstacle_polys:
            return float('inf')
        
        vehicle_poly = create_vehicle_rect(x, y, theta,
                                          self.vehicle_length,
                                          self.vehicle_width)
        
        min_dist = float('inf')
        for poly in self.obstacle_polys:
            dist = vehicle_poly.distance(poly)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist