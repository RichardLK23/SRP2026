"""
车辆运动学模型
"""
import math
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate, rotate

from config import Config  # 改为直接导入
from geometry import create_vehicle_rect, normalize_angle, angle_diff  # 改为直接导入


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
        """应用运动基元，更新车辆位姿"""
        steering = max(-self.max_steering_angle, 
                    min(self.max_steering_angle, steering))
        
        # 使用更大的步长确保移动
        actual_step = step_size * direction
        
        if abs(steering) < 1e-6:
            # 直线运动
            x_new = x + actual_step * math.cos(theta)
            y_new = y + actual_step * math.sin(theta)
            theta_new = theta
        else:
            # 圆弧运动
            radius = self.wheelbase / math.tan(steering)
            # 如果是倒车，转弯半径取反
            if direction < 0:
                radius = -radius
            
            delta_theta = step_size / radius * direction
            
            # 圆心位置
            cx = x - radius * math.sin(theta)
            cy = y + radius * math.cos(theta)
            
            # 旋转后的新位置
            x_new = cx + radius * math.sin(theta + delta_theta)
            y_new = cy - radius * math.cos(theta + delta_theta)
            theta_new = normalize_angle(theta + delta_theta)
        
        # 调试输出 - 只在首次调用时打印
        if not hasattr(self, '_debug_printed'):
            print(f"运动基元: 从 ({x:.2f},{y:.2f}) 到 ({x_new:.2f},{y_new:.2f}), 步长={step_size}")
            self._debug_printed = True
        
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
    """
    x1, y1, theta1 = start
    x2, y2, theta2 = goal
    
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    if dist < 50.0:
        # 使用更小的步长，生成更多点以便碰撞检测
        small_step = min(step_size, 0.1)  # 使用0.1m步长
        num_points = max(10, int(dist / small_step))
        
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            theta = normalize_angle(theta1 + t * angle_diff(theta1, theta2))
            path.append((x, y, theta))
        
        # 打印路径信息用于调试
        print(f"生成RS路径: 起点({x1:.2f},{y1:.2f}) -> 终点({x2:.2f},{y2:.2f}), 共{len(path)}个点")
        return path
    
    return None