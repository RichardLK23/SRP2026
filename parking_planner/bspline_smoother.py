"""
模块3：路径平滑 - 使用线性插值
"""
import math
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon

from config import Config
from collision_checker import CollisionChecker
from geometry import distance, normalize_angle


class BSplineSmoother:
    """路径平滑器（使用线性插值）"""
    
    def __init__(self, config: Config, obstacle_polys: List[Polygon]):
        self.config = config
        self.collision_checker = CollisionChecker(obstacle_polys, config)
        self.max_curvature = config.bspline.MAX_CURVATURE
        self.curvature_safety = config.bspline.CURVATURE_SAFETY_FACTOR
    
    def smooth_path(self, rough_path):
        """平滑路径 - 使用线性插值（不产生振荡）"""
        if len(rough_path) < 2:
            return rough_path
        
        # 生成密集的插值路径
        num_points = max(len(rough_path) * 5, 200)
        smooth_path = self._interpolate_path(rough_path, num_points)
        
        # 对航向角做平滑
        smooth_path = self._smooth_headings(smooth_path)
        
        print(f"平滑完成！路径点数: {len(smooth_path)}")
        return smooth_path
    
    def _interpolate_path(self, path, num_points):
        """线性插值路径"""
        if len(path) < 2:
            return [(path[0][0], path[0][1], path[0][2]) for _ in range(num_points)]
        
        # 计算总长度
        segments = []
        total_length = 0
        for i in range(len(path) - 1):
            dist = distance((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]))
            segments.append(dist)
            total_length += dist
        
        if total_length < 0.001:
            return [(path[0][0], path[0][1], path[0][2]) for _ in range(num_points)]
        
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
                    theta = normalize_angle(path[j][2] + local_t * self._angle_diff(path[j][2], path[j+1][2]))
                    
                    interpolated.append((x, y, theta))
                    break
                
                cum_dist += seg_len
        
        # 确保包含最后一个点
        if len(interpolated) > 0:
            interpolated[-1] = (path[-1][0], path[-1][1], path[-1][2])
        
        return interpolated
    
    def _smooth_headings(self, path):
        """平滑航向角序列"""
        if len(path) < 3:
            return path
        
        smoothed = []
        window_size = 5
        
        for i in range(len(path)):
            x, y, theta = path[i]
            
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            
            sin_sum = 0.0
            cos_sum = 0.0
            count = 0
            
            for j in range(start_idx, end_idx):
                sin_sum += math.sin(path[j][2])
                cos_sum += math.cos(path[j][2])
                count += 1
            
            if count > 0:
                avg_theta = math.atan2(sin_sum / count, cos_sum / count)
                smoothed.append((x, y, normalize_angle(avg_theta)))
            else:
                smoothed.append((x, y, theta))
        
        return smoothed
    
    def _angle_diff(self, a1, a2):
        """计算角度差"""
        diff = (a2 - a1) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff
    
    def _calculate_curvatures(self, path):
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
            
            cross = dx1 * dy2 - dy1 * dx2
            curvature = 2 * abs(cross) / (dist1 * dist2 * (dist1 + dist2))
            curvatures.append(curvature)
        
        return curvatures if curvatures else [0.0]