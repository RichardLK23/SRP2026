"""
可视化工具
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon

from config import Config
from geometry import create_vehicle_rect, distance

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    try:
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        pass

# 在类加载时设置字体
setup_chinese_font()


class Visualizer:
    """可视化工具"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fig_size = config.visualization.FIGURE_SIZE
        self.save_path = config.visualization.FIGURE_SAVE_PATH
        
        self.color_obstacle = config.visualization.COLOR_OBSTACLE
        self.color_start = config.visualization.COLOR_START
        self.color_goal = config.visualization.COLOR_GOAL
        self.color_guide = config.visualization.COLOR_GUIDE_POINTS
        self.color_raw = config.visualization.COLOR_RAW_PATH
        self.color_smooth = config.visualization.COLOR_SMOOTH_PATH
        self.color_vehicle = config.visualization.COLOR_VEHICLE
        
        # 再次确保字体设置
        setup_chinese_font()
    
    def plot_scene(self, 
                   obstacles: List[Polygon],
                   start: Tuple[float, float, float],
                   goal: Tuple[float, float, float],
                   guide_points: List[Tuple[float, float, float]],
                   rough_path: List[Tuple[float, float, float]],
                   smooth_path: List[Tuple[float, float, float]]):
        """绘制完整场景"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制障碍物
        for idx, poly in enumerate(obstacles):
            try:
                x, y = poly.exterior.xy
                label = '障碍物' if idx == 0 else ''
                ax.fill(x, y, color=self.color_obstacle, alpha=0.7, label=label)
                ax.plot(x, y, 'k-', linewidth=1)
            except Exception as e:
                print(f"绘制障碍物警告: {e}")
                continue
        
        # 绘制起点和终点
        ax.plot(start[0], start[1], 'go', markersize=12, label='起点')
        self._draw_vehicle(ax, start[0], start[1], start[2], 'green')
        
        ax.plot(goal[0], goal[1], 'ro', markersize=12, label='目标点')
        self._draw_vehicle(ax, goal[0], goal[1], goal[2], 'red')
        
        # 绘制引导点
        if guide_points:
            gx = [gp[0] for gp in guide_points]
            gy = [gp[1] for gp in guide_points]
            ax.plot(gx, gy, 'b-o', linewidth=2, markersize=8, label='引导点')
            for gp in guide_points:
                self._draw_arrow(ax, gp[0], gp[1], gp[2], 0.3, 'blue')
        
        # 绘制粗糙路径
        if rough_path:
            rx = [p[0] for p in rough_path]
            ry = [p[1] for p in rough_path]
            ax.plot(rx, ry, 'orange', linewidth=2, linestyle='--', label='粗糙路径')
            # 绘制路径上的车辆姿态
            step = max(1, len(rough_path)//20)
            for i in range(0, len(rough_path), step):
                p = rough_path[i]
                self._draw_vehicle(ax, p[0], p[1], p[2], 'orange', alpha=0.3)
        
        # 绘制平滑路径
        if smooth_path:
            sx = [p[0] for p in smooth_path]
            sy = [p[1] for p in smooth_path]
            ax.plot(sx, sy, 'purple', linewidth=3, label='平滑路径')
            step = max(1, len(smooth_path)//30)
            for i in range(0, len(smooth_path), step):
                p = smooth_path[i]
                self._draw_vehicle(ax, p[0], p[1], p[2], 'purple', alpha=0.2)
        
        # 设置标签 - 使用英文避免中文显示问题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Automatic Parking Path Planning')
        ax.legend(loc='upper right')
        ax.axis('equal')
        
        # 自动调整坐标范围
        all_x = []
        all_y = []
        for poly in obstacles:
            try:
                bounds = poly.bounds
                all_x.extend([bounds[0], bounds[2]])
                all_y.extend([bounds[1], bounds[3]])
            except:
                pass
        
        if smooth_path:
            all_x.extend([p[0] for p in smooth_path])
            all_y.extend([p[1] for p in smooth_path])
        
        if all_x and all_y:
            margin = 2.0
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        if self.config.visualization.SAVE_FIGURE:
            plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存至: {self.save_path}")
    
    def _draw_vehicle(self, ax, x: float, y: float, heading: float, 
                      color: str, alpha: float = 1.0):
        """绘制车辆矩形"""
        length = self.config.vehicle.LENGTH
        width = self.config.vehicle.WIDTH
        
        vehicle_poly = create_vehicle_rect(x, y, heading, length, width)
        x_coords, y_coords = vehicle_poly.exterior.xy
        
        ax.fill(x_coords, y_coords, color=color, alpha=alpha * 0.3, 
                edgecolor=color, linewidth=1)
    
    def _draw_arrow(self, ax, x: float, y: float, heading: float, 
                    length: float, color: str):
        """绘制方向箭头"""
        dx = length * np.cos(heading)
        dy = length * np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, 
                fc=color, ec=color, alpha=0.7)