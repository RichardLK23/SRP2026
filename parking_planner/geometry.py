"""
几何计算工具函数
"""
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box, GeometryCollection
from shapely.affinity import rotate, translate
from shapely.strtree import STRtree
from typing import List, Tuple, Optional


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点欧氏距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_diff(a1: float, a2: float) -> float:
    """计算两个角度之间的最小差值（弧度）"""
    diff = (a2 - a1) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff


def normalize_angle(theta: float) -> float:
    """归一化角度到 [-pi, pi]"""
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    return theta


def point_to_segment_distance(p: Tuple[float, float], 
                              p1: Tuple[float, float], 
                              p2: Tuple[float, float]) -> float:
    """点到线段的最短距离"""
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return distance(p, p1)
    
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return distance(p, (proj_x, proj_y))


def line_intersects_polygon(p1: Tuple[float, float], 
                            p2: Tuple[float, float], 
                            polygon: Polygon) -> bool:
    """检查线段是否与多边形相交"""
    line = LineString([p1, p2])
    if line.intersects(polygon):
        return True
    return False


def create_vehicle_rect(x: float, y: float, heading: float, 
                        length: float, width: float) -> Polygon:
    """
    创建车辆矩形 - 强制返回有效的 Polygon
    """
    try:
        # 确保所有参数都是float
        x = float(x)
        y = float(y)
        heading = float(heading)
        length = float(length)
        width = float(width)
        
        # 创建矩形（中心在原点）
        half_l = length / 2.0
        half_w = width / 2.0
        
        # 直接构造矩形顶点，避免使用 box + rotate 可能产生的问题
        # 矩形的四个角（未旋转）
        corners = [
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w)
        ]
        
        # 旋转和平移
        cos_theta = math.cos(heading)
        sin_theta = math.sin(heading)
        
        transformed_corners = []
        for cx, cy in corners:
            # 旋转
            rx = cx * cos_theta - cy * sin_theta
            ry = cx * sin_theta + cy * cos_theta
            # 平移
            tx = rx + x
            ty = ry + y
            transformed_corners.append((tx, ty))
        
        # 创建多边形
        vehicle_poly = Polygon(transformed_corners)
        
        # 确保有效
        if vehicle_poly.is_valid and not vehicle_poly.is_empty:
            return vehicle_poly
        
        # 如果无效，尝试修复
        vehicle_poly = vehicle_poly.buffer(0)
        if vehicle_poly.is_valid and not vehicle_poly.is_empty:
            # 如果 buffer 返回的是 Polygon，直接返回
            if isinstance(vehicle_poly, Polygon):
                return vehicle_poly
            # 如果是 GeometryCollection，取第一个 Polygon
            elif isinstance(vehicle_poly, GeometryCollection):
                for geom in vehicle_poly.geoms:
                    if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                        return geom
        
        # 最后的后备：返回一个小圆形
        print(f"警告: 创建车辆矩形失败，使用后备圆形，位置 ({x:.2f}, {y:.2f})")
        return Point(x, y).buffer(0.5)
                
    except Exception as e:
        print(f"创建车辆矩形错误: {e}")
        return Point(x, y).buffer(0.5)


def compute_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算从p1指向p2的方向角"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)


def is_point_in_polygon(point: Tuple[float, float], polygon: Polygon) -> bool:
    """检查点是否在多边形内部"""
    return Point(point).within(polygon)


def build_spatial_index(polygons: List[Polygon]) -> STRtree:
    """构建空间索引加速碰撞检测"""
    valid_polygons = [p for p in polygons if p is not None and isinstance(p, Polygon) and p.is_valid and not p.is_empty]
    return STRtree(valid_polygons) if valid_polygons else None


def polygon_from_points(points: List[Tuple[float, float]]) -> Polygon:
    """从点列表创建多边形"""
    if len(points) < 3:
        return None
    return Polygon(points)


def simplify_polygon(polygon: Polygon, tolerance: float = 0.1) -> Polygon:
    """简化多边形"""
    return polygon.simplify(tolerance, preserve_topology=True)