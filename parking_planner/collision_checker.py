"""
碰撞检测模块
"""
from typing import List, Tuple, Union
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, GeometryCollection
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
            if p is None:
                continue
            
            if isinstance(p, Polygon):
                if p.is_valid and not p.is_empty:
                    self.obstacle_polys.append(p)
                else:
                    try:
                        fixed = p.buffer(0)
                        if isinstance(fixed, Polygon) and fixed.is_valid and not fixed.is_empty:
                            self.obstacle_polys.append(fixed)
                    except:
                        pass
            elif isinstance(p, MultiPolygon):
                for geom in p.geoms:
                    if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                        self.obstacle_polys.append(geom)
            elif isinstance(p, GeometryCollection):
                for geom in p.geoms:
                    if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                        self.obstacle_polys.append(geom)
            else:
                try:
                    if hasattr(p, 'convex_hull'):
                        hull = p.convex_hull
                        if isinstance(hull, Polygon) and hull.is_valid and not hull.is_empty:
                            self.obstacle_polys.append(hull)
                except:
                    pass
        
        print(f"碰撞检测器初始化: {len(self.obstacle_polys)} 个障碍物")
        for i, poly in enumerate(self.obstacle_polys):
            try:
                bounds = poly.bounds
                print(f"  障碍物 {i}: 范围 ({bounds[0]:.1f}, {bounds[1]:.1f}) - ({bounds[2]:.1f}, {bounds[3]:.1f})")
            except:
                print(f"  障碍物 {i}: 无效")
        
        # 构建空间索引 - 只包含有效的 Polygon
        valid_polys = [p for p in self.obstacle_polys if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
        self.spatial_index = STRtree(valid_polys) if valid_polys else None
        
        self.vehicle_length = config.vehicle.LENGTH
        self.vehicle_width = config.vehicle.WIDTH
        self.clearance = config.cost.OBSTACLE_CLEARANCE
        
        print(f"车辆尺寸: {self.vehicle_length}m x {self.vehicle_width}m, 安全间隙: {self.clearance}m")
        
        self.debug_count = 0
        self.collision_count = 0
    
    def check_vehicle_collision(self, x: float, y: float, theta: float) -> bool:
        """检查车辆在当前位置是否碰撞"""
        if not self.obstacle_polys:
            return False
        
        try:
            # 创建车辆矩形
            vehicle_poly = create_vehicle_rect(x, y, theta, 
                                              self.vehicle_length, 
                                              self.vehicle_width)
            
            if vehicle_poly is None:
                return False
            if not isinstance(vehicle_poly, Polygon):
                return False
            if not vehicle_poly.is_valid or vehicle_poly.is_empty:
                return False
            
            # 添加安全间隙
            check_poly = vehicle_poly
            if self.clearance > 0:
                check_poly = vehicle_poly.buffer(self.clearance)
                if not isinstance(check_poly, Polygon):
                    check_poly = vehicle_poly
            
            # ===== 核心修复：直接遍历所有障碍物进行相交检测 =====
            # 先尝试使用空间索引
            if self.spatial_index:
                try:
                    # 使用 intersects 方法查询
                    candidates = self.spatial_index.query(check_poly)
                    for poly in candidates:
                        if poly is None:
                            continue
                        if not isinstance(poly, Polygon):
                            continue
                        # 使用 intersects 方法
                        if check_poly.intersects(poly):
                            self.collision_count += 1
                            if self.collision_count <= 5:
                                print(f"🚗 车辆碰撞! 位置 ({x:.2f}, {y:.2f}), 障碍物: {poly.bounds}")
                            return True
                except Exception as e:
                    if self.debug_count < 3:
                        print(f"空间索引查询错误: {e}")
                        self.debug_count += 1
            
            # ===== 后备方案：直接遍历所有障碍物 =====
            for poly in self.obstacle_polys:
                if poly is None:
                    continue
                if not isinstance(poly, Polygon):
                    continue
                try:
                    if check_poly.intersects(poly):
                        self.collision_count += 1
                        if self.collision_count <= 5:
                            print(f"🚗 车辆碰撞! 位置 ({x:.2f}, {y:.2f}), 障碍物: {poly.bounds}")
                        return True
                except Exception as e:
                    if self.debug_count < 3:
                        print(f"相交检测错误: {e}")
                        self.debug_count += 1
                    continue
                        
        except Exception as e:
            if self.debug_count < 3:
                print(f"车辆碰撞检测错误: {e}")
                self.debug_count += 1
            pass
        
        return False
    
    def check_path_collision(self, path: List[Tuple[float, float, float]]) -> bool:
        """检查路径上所有点是否碰撞"""
        if not path:
            return False
        
        self.collision_count = 0
        
        print(f"检查路径碰撞，共 {len(path)} 个点...")
        
        for i, (x, y, theta) in enumerate(path):
            if self.check_vehicle_collision(x, y, theta):
                print(f"❌ 路径点 {i} 碰撞: ({x:.2f}, {y:.2f})")
                return True
            
            if i % 50 == 0 and i > 0:
                print(f"  已检查 {i} 个点，安全")
        
        print(f"✅ 所有 {len(path)} 个点都安全")
        return False
    
    def check_line_collision(self, p1: Tuple[float, float], 
                             p2: Tuple[float, float]) -> bool:
        """检查线段是否与障碍物碰撞"""
        if not self.obstacle_polys:
            return False
        
        try:
            p1 = (float(p1[0]), float(p1[1]))
            p2 = (float(p2[0]), float(p2[1]))
        except:
            return False
        
        line = LineString([p1, p2])
        
        try:
            if self.spatial_index:
                candidates = self.spatial_index.query(line)
                for poly in candidates:
                    if poly is None:
                        continue
                    if not isinstance(poly, Polygon):
                        continue
                    if line.intersects(poly):
                        return True
            else:
                for poly in self.obstacle_polys:
                    if poly is not None and isinstance(poly, Polygon):
                        if line.intersects(poly):
                            return True
        except Exception as e:
            pass
        
        return False
    
    def get_clearance_to_obstacles(self, x: float, y: float, theta: float) -> float:
        """获取车辆到最近障碍物的距离"""
        if not self.obstacle_polys:
            return 10.0
        
        try:
            vehicle_poly = create_vehicle_rect(x, y, theta,
                                              self.vehicle_length,
                                              self.vehicle_width)
            
            if vehicle_poly is None or not isinstance(vehicle_poly, Polygon):
                return 10.0
            if not vehicle_poly.is_valid or vehicle_poly.is_empty:
                return 10.0
            
            min_dist = float('inf')
            for poly in self.obstacle_polys:
                if poly is not None and isinstance(poly, Polygon):
                    dist = vehicle_poly.distance(poly)
                    if dist < min_dist:
                        min_dist = dist
            return min_dist if min_dist != float('inf') else 10.0
        except:
            return 10.0