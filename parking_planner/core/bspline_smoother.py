"""
模块3：B样条平滑与后处理
"""
import math  # 添加math导入
import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon

from ..config import Config
from .collision_checker import CollisionChecker
from ...utils.geometry import distance, normalize_angle  # 添加distance导入


class BSplineSmoother:
    """B样条路径平滑器"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        self.collision_checker = CollisionChecker(obstacle_polys, config)
        
        # B样条参数
        self.degree = config.bspline.BSPLINE_DEGREE
        self.sample_interval = config.bspline.CONTROL_POINT_SAMPLE_INTERVAL
        self.max_iterations = config.bspline.MAX_SMOOTHING_ITERATIONS
        self.smoothing_weight = config.bspline.SMOOTHING_WEIGHT
        self.max_curvature = config.bspline.MAX_CURVATURE
        self.curvature_safety = config.bspline.CURVATURE_SAFETY_FACTOR
        
        # 碰撞校验参数
        self.collision_resolution = config.bspline.COLLISION_CHECK_RESOLUTION
        self.max_collision_iter = config.bspline.MAX_COLLISION_ITERATIONS
    
    def smooth_path(self, 
                   rough_path: List[Tuple[float, float, float]]) -> Optional[List[Tuple[float, float, float]]]:
        """
        平滑路径
        
        Args:
            rough_path: 粗糙路径位姿序列
        
        Returns:
            平滑路径位姿序列
        """
        if len(rough_path) < 4:
            print("路径点太少，无法进行B样条平滑")
            return rough_path
        
        # 1. 提取控制点
        control_points = self._extract_control_points(rough_path)
        
        # 2. B样条拟合
        smooth_path = self._fit_bspline(control_points, len(rough_path) * 3)
        
        # 3. 碰撞校验与迭代
        for iteration in range(self.max_collision_iter):
            # 检查碰撞
            has_collision = self.collision_checker.check_path_collision(smooth_path)
            
            if not has_collision:
                # 检查曲率
                curvatures = self._calculate_curvatures(smooth_path)
                max_curv = max(abs(c) for c in curvatures) if curvatures else 0
                
                if max_curv <= self.max_curvature * self.curvature_safety:
                    print(f"平滑成功！迭代次数: {iteration+1}, 最大曲率: {max_curv:.3f}")
                    return smooth_path
                else:
                    print(f"曲率过大: {max_curv:.3f}，尝试重新平滑")
            
            # 减小采样间隔重新拟合
            self.sample_interval = max(1, self.sample_interval - 1)
            control_points = self._extract_control_points(rough_path)
            smooth_path = self._fit_bspline(control_points, len(rough_path) * 4)
        
        print(f"平滑迭代达到最大次数 ({self.max_collision_iter})")
        return smooth_path
    
    def _extract_control_points(self, 
                               path: List[Tuple[float, float, float]]) -> np.ndarray:
        """提取控制点"""
        # 只取x, y坐标
        points = [(p[0], p[1]) for p in path]
        
        # 均匀采样
        if self.sample_interval <= 1:
            return np.array(points)
        
        sampled = []
        for i in range(0, len(points), self.sample_interval):
            sampled.append(points[i])
        
        # 确保包含最后一个点
        if len(sampled) > 0 and sampled[-1] != points[-1]:
            sampled.append(points[-1])
        
        return np.array(sampled)
    
    def _fit_bspline(self, 
                    control_points: np.ndarray, 
                    num_points: int) -> List[Tuple[float, float, float]]:
        """拟合B样条曲线"""
        if len(control_points) < 4:
            # 如果控制点太少，直接返回原始路径
            return [(p[0], p[1], 0.0) for p in control_points]
        
        try:
            # 提取x和y坐标
            x = control_points[:, 0]
            y = control_points[:, 1]
            
            # B样条拟合
            tck, u = splprep([x, y], s=self.smoothing_weight, k=self.degree)
            
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
            print(f"B样条拟合失败: {e}")
            # 返回原始路径的插值
            return self._interpolate_path(control_points, num_points)
    
    def _interpolate_path(self, 
                         points: np.ndarray, 
                         num_points: int) -> List[Tuple[float, float, float]]:
        """线性插值路径（备用方案）"""
        if len(points) < 2:
            return [(points[0][0], points[0][1], 0.0)]
        
        interpolated = []
        total_length = 0
        segments = []
        
        for i in range(len(points) - 1):
            dist = distance(points[i], points[i+1])
            segments.append(dist)
            total_length += dist
        
        if total_length == 0:
            return [(points[0][0], points[0][1], 0.0)]
        
        # 均匀采样
        for i in range(num_points):
            t = i / (num_points - 1)
            target_dist = t * total_length
            
            cum_dist = 0
            for j, seg_len in enumerate(segments):
                if cum_dist + seg_len >= target_dist or j == len(segments) - 1:
                    # 插值
                    local_t = (target_dist - cum_dist) / seg_len if seg_len > 0 else 0
                    local_t = max(0, min(1, local_t))
                    
                    x = points[j][0] + local_t * (points[j+1][0] - points[j][0])
                    y = points[j][1] + local_t * (points[j+1][1] - points[j][1])
                    
                    # 计算航向
                    heading = math.atan2(points[j+1][1] - points[j][1],
                                        points[j+1][0] - points[j][0])
                    
                    interpolated.append((x, y, normalize_angle(heading)))
                    break
                
                cum_dist += seg_len
        
        return interpolated
    
    def _calculate_curvatures(self, 
                             path: List[Tuple[float, float, float]]) -> List[float]:
        """计算路径各点的曲率"""
        if len(path) < 3:
            return []
        
        curvatures = []
        
        for i in range(1, len(path) - 1):
            p0 = path[i-1]
            p1 = path[i]
            p2 = path[i+1]
            
            # 使用三点法估算曲率
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            
            # 计算叉积和点积
            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2
            
            # 计算曲率
            if dx1 == 0 and dy1 == 0 or dx2 == 0 and dy2 == 0:
                curvatures.append(0.0)
                continue
            
            # 曲率 = 2 * sin(角度差) / (边长之和)
            dist1 = math.sqrt(dx1*dx1 + dy1*dy1)
            dist2 = math.sqrt(dx2*dx2 + dy2*dy2)
            
            if dist1 == 0 or dist2 == 0:
                curvatures.append(0.0)
                continue
            
            # 近似曲率
            curvature = 2 * abs(cross) / (dist1 * dist2 * (dist1 + dist2))
            curvatures.append(curvature)
        
        return curvatures
    
    def check_path_curvature(self, 
                            path: List[Tuple[float, float, float]]) -> bool:
        """检查路径曲率是否满足要求"""
        curvatures = self._calculate_curvatures(path)
        if not curvatures:
            return True
        
        max_curv = max(abs(c) for c in curvatures)
        return max_curv <= self.max_curvature * self.curvature_safety