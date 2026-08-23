"""
模块3：B样条平滑与后处理
"""
import math
import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon

from config import Config
from collision_checker import CollisionChecker
from geometry import distance, normalize_angle


class BSplineSmoother:
    """B样条路径平滑器"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        self.collision_checker = CollisionChecker(obstacle_polys, config)
        
        self.degree = config.bspline.BSPLINE_DEGREE
        self.sample_interval = config.bspline.CONTROL_POINT_SAMPLE_INTERVAL
        self.max_iterations = config.bspline.MAX_SMOOTHING_ITERATIONS
        self.smoothing_weight = config.bspline.SMOOTHING_WEIGHT
        self.max_curvature = config.bspline.MAX_CURVATURE
        self.curvature_safety = config.bspline.CURVATURE_SAFETY_FACTOR
        
        self.collision_resolution = config.bspline.COLLISION_CHECK_RESOLUTION
        self.max_collision_iter = config.bspline.MAX_COLLISION_ITERATIONS
    
    def smooth_path(self, 
                   rough_path: List[Tuple[float, float, float]]) -> Optional[List[Tuple[float, float, float]]]:
        """平滑路径"""
        if len(rough_path) < 3:
            print(f"路径点太少 ({len(rough_path)}个)，进行插值")
            return self._interpolate_path(rough_path, 100)
        
        # 提取控制点 - 使用更密集的采样
        control_points = self._extract_control_points(rough_path)
        
        # 如果控制点太少，使用所有点
        if len(control_points) < 4:
            print("控制点太少，使用所有路径点")
            control_points = np.array([(p[0], p[1]) for p in rough_path])
        
        # 生成足够多的点
        num_points = max(len(rough_path) * 3, 150)
        
        # 尝试B样条拟合
        smooth_path = self._fit_bspline(control_points, num_points)
        
        # 如果B样条失败，使用线性插值
        if smooth_path is None or len(smooth_path) < 10:
            print("B样条拟合失败，使用线性插值")
            smooth_path = self._interpolate_path(rough_path, num_points)
        
        # 碰撞校验与迭代
        for iteration in range(self.max_collision_iter):
            has_collision = self.collision_checker.check_path_collision(smooth_path)
            
            if not has_collision:
                curvatures = self._calculate_curvatures(smooth_path)
                max_curv = max(abs(c) for c in curvatures) if curvatures else 0
                
                if max_curv <= self.max_curvature * self.curvature_safety:
                    print(f"平滑成功！迭代次数: {iteration+1}, 最大曲率: {max_curv:.4f}, 路径点数: {len(smooth_path)}")
                    return smooth_path
                else:
                    print(f"曲率过大: {max_curv:.4f}，尝试重新平滑")
            
            # 减小采样间隔重新拟合
            self.sample_interval = max(1, self.sample_interval - 1)
            control_points = self._extract_control_points(rough_path)
            smooth_path = self._fit_bspline(control_points, num_points * 2)
            if smooth_path is None:
                smooth_path = self._interpolate_path(rough_path, num_points * 2)
        
        print(f"平滑迭代达到最大次数 ({self.max_collision_iter})，返回当前路径")
        return smooth_path
    
    def _extract_control_points(self, 
                               path: List[Tuple[float, float, float]]) -> np.ndarray:
        """提取控制点"""
        points = [(p[0], p[1]) for p in path]
        
        # 使用更小的采样间隔
        interval = max(1, self.sample_interval // 2)
        
        sampled = []
        for i in range(0, len(points), interval):
            sampled.append(points[i])
        
        # 确保包含最后一个点
        if len(sampled) > 0 and sampled[-1] != points[-1]:
            sampled.append(points[-1])
        
        # 如果采样点太少，直接使用所有点
        if len(sampled) < 4:
            return np.array(points)
        
        return np.array(sampled)
    
    def _fit_bspline(self, 
                    control_points: np.ndarray, 
                    num_points: int) -> Optional[List[Tuple[float, float, float]]]:
        """拟合B样条曲线"""
        if len(control_points) < 4:
            return None
        
        try:
            # 提取x和y坐标
            x = control_points[:, 0]
            y = control_points[:, 1]
            
            # 检查是否有重复点
            unique_points = []
            for i, (xi, yi) in enumerate(zip(x, y)):
                if i == 0:
                    unique_points.append((xi, yi))
                else:
                    if distance((xi, yi), unique_points[-1]) > 0.001:
                        unique_points.append((xi, yi))
            
            if len(unique_points) < 4:
                return None
            
            x = np.array([p[0] for p in unique_points])
            y = np.array([p[1] for p in unique_points])
            
            # 使用较小的平滑参数
            s = min(self.smoothing_weight, 0.1)
            
            # B样条拟合
            tck, u = splprep([x, y], s=s, k=min(3, len(unique_points)-1))
            
            # 生成密集点
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            
            # 计算航向角
            dx, dy = splev(u_new, tck, der=1)
            headings = np.arctan2(dy, dx)
            
            # 组合结果
            smooth_path = []
            for i in range(len(x_new)):
                theta = normalize_angle(headings[i])
                smooth_path.append((float(x_new[i]), float(y_new[i]), theta))
            
            return smooth_path
            
        except Exception as e:
            print(f"B样条拟合详细错误: {e}")
            return None
    
    def _interpolate_path(self, 
                         path: List[Tuple[float, float, float]], 
                         num_points: int) -> List[Tuple[float, float, float]]:
        """线性插值路径（备用方案）"""
        if len(path) < 2:
            return [(float(path[0][0]), float(path[0][1]), 0.0) for _ in range(num_points)]
        
        # 计算总长度
        segments = []
        total_length = 0
        for i in range(len(path) - 1):
            dist = distance((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]))
            segments.append(dist)
            total_length += dist
        
        if total_length < 0.001:
            # 所有点都在同一位置，生成直线路径
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
                    
                    # 航向角插值
                    theta = normalize_angle(path[j][2] + local_t * self._angle_diff(path[j][2], path[j+1][2]))
                    
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
            theta = normalize_angle(start[2] + t * self._angle_diff(start[2], end[2]))
            straight_path.append((x, y, theta))
        
        return straight_path
    
    def _angle_diff(self, a1: float, a2: float) -> float:
        """计算角度差"""
        diff = (a2 - a1) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff
    
    def _calculate_curvatures(self, 
                             path: List[Tuple[float, float, float]]) -> List[float]:
        """计算路径各点的曲率"""
        if len(path) < 3:
            return [0.0]
        
        curvatures = []
        
        for i in range(1, len(path) - 1):
            p0 = path[i-1]
            p1 = path[i]
            p2 = path[i+1]
            
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            
            dist1 = math.sqrt(dx1*dx1 + dy1*dy1)
            dist2 = math.sqrt(dx2*dx2 + dy2*dy2)
            
            if dist1 < 0.001 or dist2 < 0.001:
                curvatures.append(0.0)
                continue
            
            # 使用叉积计算曲率
            cross = dx1 * dy2 - dy1 * dx2
            curvature = 2 * abs(cross) / (dist1 * dist2 * (dist1 + dist2))
            curvatures.append(curvature)
        
        if not curvatures:
            return [0.0]
        
        return curvatures
    
    def check_path_curvature(self, 
                            path: List[Tuple[float, float, float]]) -> bool:
        """检查路径曲率是否满足要求"""
        curvatures = self._calculate_curvatures(path)
        if not curvatures:
            return True
        
        max_curv = max(abs(c) for c in curvatures)
        return max_curv <= self.max_curvature * self.curvature_safety