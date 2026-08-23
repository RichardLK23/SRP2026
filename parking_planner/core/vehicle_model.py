"""
车辆运动学模型
"""
import math  # 确保math已导入
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate, rotate

from ..config import Config
from ...utils.geometry import create_vehicle_rect, normalize_angle, angle_diff  # 添加angle_diff


class VehicleModel:
    """车辆运动学模型"""
    
    def __init__(self, config: Config):
        self.config = config
        self.length = config.vehicle.LENGTH
        self.width = config.vehicle.WIDTH
        self.wheelbase = config.vehicle.WHEELBASE
        self.max_steering_angle = config.vehicle.max_steering_rad
        self.turning_radius = config.vehicle.turning_radius
        self.step_size = config.vehicle.STEP_SIZE
    
    def apply_motion_primitive(self, x: float, y: float, theta: float,
                               steering: float, direction: int, 
                               step_size: float) -> Tuple[float, float, float]:
        """
        应用运动基元，更新车辆位姿
        
        Args:
            x, y, theta: 当前位置和航向
            steering: 转向角（弧度），正值左转
            direction: 1前进，-1后退
            step_size: 步长
        
        Returns:
            新的位姿 (x, y, theta)
        """
        # 限制转向角
        steering = max(-self.max_steering_angle, 
                      min(self.max_steering_angle, steering))
        
        # 使用自行车模型更新
        if abs(steering) < 1e-6:
            # 直线运动
            x_new = x + direction * step_size * math.cos(theta)
            y_new = y + direction * step_size * math.sin(theta)
            theta_new = theta
        else:
            # 圆弧运动
            radius = self.wheelbase / math.tan(steering)
            # 实际转弯半径应考虑方向
            radius = radius * direction
            delta_theta = step_size / radius
            
            # 圆心位置
            cx = x - radius * math.sin(theta)
            cy = y + radius * math.cos(theta)
            
            # 旋转后的新位置
            x_new = cx + radius * math.sin(theta + delta_theta)
            y_new = cy - radius * math.cos(theta + delta_theta)
            theta_new = normalize_angle(theta + delta_theta)
        
        return (x_new, y_new, theta_new)
    
    def get_vehicle_polygon(self, x: float, y: float, theta: float) -> Polygon:
        """获取车辆多边形"""
        return create_vehicle_rect(x, y, theta, self.length, self.width)
    
    def check_collision(self, x: float, y: float, theta: float,
                        obstacle_polys: List[Polygon]) -> bool:
        """检查当前位置是否碰撞"""
        vehicle_poly = self.get_vehicle_polygon(x, y, theta)
        for poly in obstacle_polys:
            if vehicle_poly.intersects(poly):
                return True
        return False
    
    def check_path_collision(self, path: List[Tuple[float, float, float]],
                             obstacle_polys: List[Polygon]) -> bool:
        """检查路径上所有点是否碰撞"""
        for x, y, theta in path:
            if self.check_collision(x, y, theta, obstacle_polys):
                return True
        return False


def generate_reeds_shepp_path(start: Tuple[float, float, float],
                              goal: Tuple[float, float, float],
                              turning_radius: float,
                              step_size: float) -> Optional[List[Tuple[float, float, float]]]:
    """
    生成Reeds-Shepp曲线路径（简化版本）
    
    这里实现一个简化的RS曲线生成，完整实现需要处理所有15种组合
    """
    # 简化实现：仅尝试直线+圆弧的组合
    # 实际应用中建议使用成熟的RS曲线库
    
    x1, y1, theta1 = start
    x2, y2, theta2 = goal
    
    # 检查是否可以直接用直线连接
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    if dist < 50.0:  # 距离阈值
        # 生成直线路径
        num_points = max(2, int(dist / step_size))
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            # 航向角线性插值
            theta = normalize_angle(theta1 + t * angle_diff(theta1, theta2))
            path.append((x, y, theta))
        return path
    
    return None


def angle_diff(a1: float, a2: float) -> float:
    """计算角度差"""
    diff = (a2 - a1) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff