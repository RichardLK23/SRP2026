"""
几何计算工具函数
"""
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box
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
    """创建车辆矩形"""
    rect = box(-length/2, -width/2, length/2, width/2)
    rect = rotate(rect, math.degrees(heading), origin=(0, 0), use_radians=False)
    rect = translate(rect, x, y)
    return rect


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
    return STRtree(polygons)


def polygon_from_points(points: List[Tuple[float, float]]) -> Polygon:
    """从点列表创建多边形"""
    if len(points) < 3:
        return None
    return Polygon(points)


def simplify_polygon(polygon: Polygon, tolerance: float = 0.1) -> Polygon:
    """简化多边形"""
    return polygon.simplify(tolerance, preserve_topology=True)