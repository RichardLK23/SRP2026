"""
自动泊车路径规划系统配置文件
"""
import math

# ============================================
# 1. 车辆运动学参数
# ============================================
class VehicleConfig:
    """车辆基本参数（参考：中型轿车）"""
    
    # 车辆尺寸（单位：米）
    LENGTH = 4.5          # 车长
    WIDTH = 1.8           # 车宽
    WHEELBASE = 2.7       # 轴距
    REAR_OVERHANG = 0.8   # 后悬（后轴到车尾距离）
    FRONT_OVERHANG = 1.0  # 前悬（前轴到车头距离）
    
    # 运动学限制
    MAX_STEERING_ANGLE = 35.0  # 最大前轮转角（度）
    MIN_TURNING_RADIUS = 5.0   # 最小转弯半径（米）
    
    # 速度/步长参数
    STEP_SIZE = 0.2       # 搜索步长（米）
    REVERSE_SPEED = -1.0  # 倒车速度
    FORWARD_SPEED = 1.0   # 前进速度
    
    @property
    def max_steering_rad(self):
        """最大转向角（弧度）"""
        return math.radians(self.MAX_STEERING_ANGLE)
    
    @property
    def turning_radius(self):
        """实际最小转弯半径"""
        return self.WHEELBASE / math.tan(self.max_steering_rad)


# ============================================
# 2. 混合A*搜索参数
# ============================================
class HybridAStarConfig:
    """混合A*算法搜索参数"""
    
    # 网格离散化分辨率
    XY_RESOLUTION = 0.2   # 位置网格分辨率（米）
    THETA_RESOLUTION = 5  # 航向角网格分辨率（度）
    
    # 搜索限制
    MAX_ITERATIONS = 50000      # 最大搜索迭代次数
    MAX_EXPANSION_NODES = 10000 # 最大扩展节点数
    TIME_LIMIT = 30.0           # 搜索时间限制（秒）
    
    # 运动基元设置
    MOTION_PRIMITIVES = [
        (0, 1, 1.0),                    # 直行
        (20, 1, 1.0),                   # 左转小
        (-20, 1, 1.0),                  # 右转小
        (35, 1, 1.0),                   # 左转大
        (-35, 1, 1.0),                  # 右转大
        (0, -1, 0.8),                   # 直倒
        (15, -1, 0.8),                  # 左倒小
        (-15, -1, 0.8),                 # 右倒小
        (25, -1, 0.8),                  # 左倒大
        (-25, -1, 0.8),                 # 右倒大
    ]
    
    # 引导点触发阈值
    GUIDE_POINT_ANGLE_THRESHOLD = 30.0  # 航向夹角阈值（度）
    GUIDE_POINT_REACH_RADIUS = 0.5      # 到达引导点的判定半径（米）
    
    # RS曲线参数
    RS_CURVE_MAX_LENGTH = 50.0          # RS曲线最大尝试长度（米）
    RS_CURVE_SAMPLE_RESOLUTION = 0.1    # RS曲线采样分辨率（米）


# ============================================
# 3. 代价函数权重
# ============================================
class CostConfig:
    """混合A*代价函数权重系数"""
    
    # 实际代价 g(n) 权重
    WEIGHT_PATH_LENGTH = 1.0       # 路径长度代价权重
    WEIGHT_GEAR_CHANGE = 5.0       # 换挡惩罚权重
    WEIGHT_STEERING_CHANGE = 2.0   # 转向角变化惩罚权重
    WEIGHT_REVERSE = 3.0           # 倒车惩罚权重
    
    # 启发代价 h(n) 权重
    WEIGHT_GUIDE_DIST = 1.0        # 到引导点距离代价权重
    WEIGHT_HEADING_DIFF = 2.0      # 航向角偏差代价权重
    WEIGHT_OBSTACLE_DIST = 10.0    # 障碍物距离代价权重
    
    # 距离阈值
    OBSTACLE_CLEARANCE = 0.2       # 障碍物安全间隙（米）
    GOAL_REACHED_THRESHOLD = 0.3   # 到达目标的判定阈值（位置）
    GOAL_HEADING_THRESHOLD = 5.0   # 到达目标的航向角阈值（度）


# ============================================
# 4. 简化可视图（SVG）参数
# ============================================
class SVGConfig:
    """简化可视图构建参数"""
    
    VIEW_WINDOW_MARGIN = 5.0       # 可视窗口扩大（米）
    SIMPLIFY_TOLERANCE = 0.1       # 多边形简化容差（米）
    USE_CONVEX_HULL = True         # 是否将凹多边形转为凸包
    REMOVE_REDUNDANT_NODES = True  # 是否移除冗余节点
    MIN_EDGE_LENGTH = 0.5          # 最小边长（米）
    DIJKSTRA_MAX_NODES = 5000      # Dijkstra最大节点数限制


# ============================================
# 5. B样条平滑参数
# ============================================
class BSplineConfig:
    """B样条平滑与后处理参数"""
    
    BSPLINE_DEGREE = 3             # B样条阶数
    CONTROL_POINT_SAMPLE_INTERVAL = 3  # 控制点采样间隔
    MAX_SMOOTHING_ITERATIONS = 10  # 最大平滑迭代次数
    SMOOTHING_WEIGHT = 0.5         # 平滑权重
    CURVATURE_CONSTRAINT_WEIGHT = 0.3  # 曲率约束权重
    COLLISION_CHECK_RESOLUTION = 0.1  # 碰撞检测采样分辨率
    MAX_COLLISION_ITERATIONS = 5      # 碰撞迭代最大次数
    MAX_CURVATURE = 0.2            # 最大允许曲率
    CURVATURE_SAFETY_FACTOR = 0.8  # 曲率安全系数


# ============================================
# 6. 车位检测参数
# ============================================
class ParkingSlotConfig:
    """车位相关参数"""
    
    PARALLEL_SLOT_LENGTH = 6.0     # 平行车位长度
    PARALLEL_SLOT_WIDTH = 2.5      # 平行车位宽度
    PERPENDICULAR_SLOT_LENGTH = 5.0  # 垂直车位长度
    PERPENDICULAR_SLOT_WIDTH = 2.5   # 垂直车位宽度
    SLOT_DETECTION_TOLERANCE = 0.3  # 车位检测容差
    MIN_SLOT_WIDTH = 2.2           # 最小可泊入车位宽度


# ============================================
# 7. 可视化参数
# ============================================
class VisualizationConfig:
    """可视化调试参数"""
    
    ENABLE_VISUALIZATION = True    # 是否启用可视化
    FIGURE_SIZE = (12, 10)         # 图形窗口大小
    SAVE_FIGURE = False            # 是否保存图像
    FIGURE_SAVE_PATH = "./output/planner_result.png"
    
    COLOR_OBSTACLE = 'gray'
    COLOR_START = 'green'
    COLOR_GOAL = 'red'
    COLOR_GUIDE_POINTS = 'blue'
    COLOR_RAW_PATH = 'orange'
    COLOR_SMOOTH_PATH = 'purple'
    COLOR_VEHICLE = 'cyan'
    COLOR_SEARCH_TREE = 'lightgray'


# ============================================
# 8. 日志与调试参数
# ============================================
class DebugConfig:
    """调试和日志参数"""
    
    LOG_LEVEL = 'INFO'
    ENABLE_PROFILING = False
    SAVE_SEARCH_TREE = False
    PRINT_PATH_INFO = True
    TEST_SCENARIO = 1


# ============================================
# 9. 主配置聚合类
# ============================================
class Config:
    """所有配置的聚合类"""
    
    def __init__(self):
        self.vehicle = VehicleConfig()
        self.hybrid_astar = HybridAStarConfig()
        self.cost = CostConfig()
        self.svg = SVGConfig()
        self.bspline = BSplineConfig()
        self.parking = ParkingSlotConfig()
        self.visualization = VisualizationConfig()
        self.debug = DebugConfig()
    
    def print_summary(self):
        """打印配置摘要"""
        print("=" * 50)
        print("自动泊车路径规划系统配置")
        print("=" * 50)
        print(f"车辆尺寸: {self.vehicle.LENGTH}m x {self.vehicle.WIDTH}m")
        print(f"最小转弯半径: {self.vehicle.turning_radius:.2f}m")
        print(f"搜索分辨率: {self.hybrid_astar.XY_RESOLUTION}m, {self.hybrid_astar.THETA_RESOLUTION}°")
        print(f"最大迭代次数: {self.hybrid_astar.MAX_ITERATIONS}")
        print(f"B样条阶数: {self.bspline.BSPLINE_DEGREE}")
        print("=" * 50)


def get_default_config():
    """获取默认配置实例"""
    return Config()


def validate_config(config):
    """验证配置参数是否合理"""
    min_turn = config.vehicle.turning_radius
    if min_turn < 2.0:
        return False, f"最小转弯半径过小: {min_turn:.2f}m"
    
    if config.hybrid_astar.XY_RESOLUTION > config.vehicle.STEP_SIZE:
        return False, f"搜索分辨率不能大于步长"
    
    if config.parking.MIN_SLOT_WIDTH < config.vehicle.WIDTH + 0.3:
        return False, f"最小车位宽度应大于车宽+安全间隙"
    
    return True, "配置有效"