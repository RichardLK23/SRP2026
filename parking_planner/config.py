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
    MIN_TURNING_RADIUS = 5.0   # 最小转弯半径（米），由轴距和最大转向角计算
    # 实际计算：MIN_TURNING_RADIUS = WHEELBASE / tan(MAX_STEERING_ANGLE_rad)
    
    # 速度/步长参数
    STEP_SIZE = 0.2       # 搜索步长（米），即每个基元的长度
    REVERSE_SPEED = -1.0  # 倒车速度（m/s，负值表示后退）
    FORWARD_SPEED = 1.0   # 前进速度（m/s）
    
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
    # 注意：航向角会被离散化为 360/THETA_RESOLUTION 个区间
    
    # 搜索限制
    MAX_ITERATIONS = 50000      # 最大搜索迭代次数
    MAX_EXPANSION_NODES = 10000 # 最大扩展节点数
    TIME_LIMIT = 30.0           # 搜索时间限制（秒）
    
    # 运动基元设置
    # 每个基元由 [转向角(度), 方向(1前进/-1后退), 步长(米)] 组成
    # 转向角为 0, ±MAX_STEERING_ANGLE/2, ±MAX_STEERING_ANGLE
    MOTION_PRIMITIVES = [
        # 前进基元
        (0, 1, 1.0),                    # 直行
        (20, 1, 1.0),                   # 左转小
        (-20, 1, 1.0),                  # 右转小
        (35, 1, 1.0),                   # 左转大
        (-35, 1, 1.0),                  # 右转大
        # 后退基元（仅在需要倒车时启用）
        (0, -1, 0.8),                   # 直倒
        (15, -1, 0.8),                  # 左倒小
        (-15, -1, 0.8),                 # 右倒小
        (25, -1, 0.8),                  # 左倒大
        (-25, -1, 0.8),                 # 右倒大
    ]
    
    # 引导点触发阈值
    GUIDE_POINT_ANGLE_THRESHOLD = 30.0  # 航向夹角阈值（度），小于此值只启用前进基元
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
    WEIGHT_GEAR_CHANGE = 5.0       # 换挡惩罚权重（每次换挡）
    WEIGHT_STEERING_CHANGE = 2.0   # 转向角变化惩罚权重（度）
    WEIGHT_REVERSE = 3.0           # 倒车惩罚权重（鼓励前进）
    
    # 启发代价 h(n) 权重
    WEIGHT_GUIDE_DIST = 1.0        # 到引导点距离代价权重
    WEIGHT_HEADING_DIFF = 2.0      # 航向角偏差代价权重（度）
    WEIGHT_OBSTACLE_DIST = 10.0    # 障碍物距离代价权重（靠近障碍物惩罚）
    
    # 距离阈值
    OBSTACLE_CLEARANCE = 0.2       # 障碍物安全间隙（米）
    GOAL_REACHED_THRESHOLD = 0.3   # 到达目标的判定阈值（位置，米）
    GOAL_HEADING_THRESHOLD = 5.0   # 到达目标的航向角阈值（度）


# ============================================
# 4. 简化可视图（SVG）参数
# ============================================
class SVGConfig:
    """简化可视图构建参数"""
    
    # 可视窗口扩展系数
    VIEW_WINDOW_MARGIN = 5.0       # 可视窗口比起点-目标区域扩大（米）
    
    # 障碍物简化参数
    SIMPLIFY_TOLERANCE = 0.1       # 多边形简化容差（米），使用Douglas-Peucker算法
    USE_CONVEX_HULL = True         # 是否将凹多边形转为凸包
    
    # 节点筛选
    REMOVE_REDUNDANT_NODES = True  # 是否移除冗余节点（共线的中间点）
    MIN_EDGE_LENGTH = 0.5          # 最小边长（米），小于此值的边将被合并
    
    # Dijkstra搜索
    DIJKSTRA_MAX_NODES = 5000      # Dijkstra最大节点数限制


# ============================================
# 5. B样条平滑参数
# ============================================
class BSplineConfig:
    """B样条平滑与后处理参数"""
    
    # B样条参数
    BSPLINE_DEGREE = 3             # B样条阶数（3为三次B样条）
    CONTROL_POINT_SAMPLE_INTERVAL = 3  # 控制点采样间隔（每隔N个路径点取一个控制点）
    
    # 平滑迭代
    MAX_SMOOTHING_ITERATIONS = 10  # 最大平滑迭代次数
    SMOOTHING_WEIGHT = 0.5         # 平滑权重（0-1之间，越大越平滑但可能偏离原路径）
    CURVATURE_CONSTRAINT_WEIGHT = 0.3  # 曲率约束权重
    
    # 碰撞校验
    COLLISION_CHECK_RESOLUTION = 0.1  # 碰撞检测采样分辨率（米）
    MAX_COLLISION_ITERATIONS = 5      # 碰撞迭代最大次数（重新采样控制点）
    
    # 曲率限制
    MAX_CURVATURE = 0.2            # 最大允许曲率（1/米），对应转弯半径5米
    CURVATURE_SAFETY_FACTOR = 0.8  # 曲率安全系数（留余量）


# ============================================
# 6. 车位检测参数
# ============================================
class ParkingSlotConfig:
    """车位相关参数"""
    
    # 标准车位尺寸（单位：米）
    PARALLEL_SLOT_LENGTH = 6.0     # 平行车位长度
    PARALLEL_SLOT_WIDTH = 2.5      # 平行车位宽度
    PERPENDICULAR_SLOT_LENGTH = 5.0  # 垂直车位长度
    PERPENDICULAR_SLOT_WIDTH = 2.5   # 垂直车位宽度
    
    # 车位检测容差
    SLOT_DETECTION_TOLERANCE = 0.3  # 车位检测容差（米）
    MIN_SLOT_WIDTH = 2.2           # 最小可泊入车位宽度（米）


# ============================================
# 7. 可视化参数
# ============================================
class VisualizationConfig:
    """可视化调试参数"""
    
    ENABLE_VISUALIZATION = True    # 是否启用可视化
    FIGURE_SIZE = (12, 10)         # 图形窗口大小
    SAVE_FIGURE = False            # 是否保存图像
    FIGURE_SAVE_PATH = "./output/planner_result.png"  # 图像保存路径
    
    # 颜色配置
    COLOR_OBSTACLE = 'gray'        # 障碍物颜色
    COLOR_START = 'green'          # 起点颜色
    COLOR_GOAL = 'red'             # 目标点颜色
    COLOR_GUIDE_POINTS = 'blue'    # 引导点颜色
    COLOR_RAW_PATH = 'orange'      # 原始搜索路径颜色
    COLOR_SMOOTH_PATH = 'purple'   # 平滑后路径颜色
    COLOR_VEHICLE = 'cyan'         # 车辆显示颜色
    COLOR_SEARCH_TREE = 'lightgray' # 搜索树显示颜色


# ============================================
# 8. 日志与调试参数
# ============================================
class DebugConfig:
    """调试和日志参数"""
    
    LOG_LEVEL = 'INFO'             # 日志级别: DEBUG, INFO, WARNING, ERROR
    ENABLE_PROFILING = False       # 是否启用性能分析
    SAVE_SEARCH_TREE = False       # 是否保存搜索树数据
    PRINT_PATH_INFO = True         # 是否打印路径信息
    
    # 测试场景选择
    TEST_SCENARIO = 1              # 1: 简单场景, 2: 复杂场景, 3: 狭窄车位


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


# ============================================
# 10. 便捷函数
# ============================================
def get_default_config():
    """获取默认配置实例"""
    return Config()


def get_config_for_scenario(scenario_id):
    """
    根据不同场景返回调整后的配置
    
    Args:
        scenario_id: 1-简单, 2-中等, 3-复杂/狭窄
    
    Returns:
        Config: 调整后的配置对象
    """
    config = Config()
    
    if scenario_id == 1:  # 简单场景（开阔区域）
        config.hybrid_astar.MAX_ITERATIONS = 20000
        config.cost.WEIGHT_REVERSE = 5.0  # 尽量不倒车
        
    elif scenario_id == 2:  # 中等场景
        config.hybrid_astar.MAX_ITERATIONS = 40000
        config.cost.WEIGHT_REVERSE = 2.0
        config.vehicle.MIN_TURNING_RADIUS = 4.5  # 更灵活
        
    elif scenario_id == 3:  # 狭窄车位场景
        config.hybrid_astar.MAX_ITERATIONS = 60000
        config.cost.WEIGHT_REVERSE = 1.0  # 允许更多倒车
        config.cost.WEIGHT_GEAR_CHANGE = 3.0
        config.vehicle.MIN_TURNING_RADIUS = 4.0
        config.parking.MIN_SLOT_WIDTH = 2.5
        
    return config


# ============================================
# 11. 配置验证函数
# ============================================
def validate_config(config):
    """
    验证配置参数是否合理
    
    Args:
        config: Config对象
    
    Returns:
        bool: 配置是否有效
        str: 错误信息（如果有）
    """
    # 检查最小转弯半径是否合理
    min_turn = config.vehicle.turning_radius
    if min_turn < 2.0:
        return False, f"最小转弯半径过小: {min_turn:.2f}m（实际车辆通常>4m）"
    
    # 检查步长是否合理
    if config.hybrid_astar.XY_RESOLUTION > config.vehicle.STEP_SIZE:
        return False, f"搜索分辨率({config.hybrid_astar.XY_RESOLUTION}m)不能大于步长({config.vehicle.STEP_SIZE}m)"
    
    # 检查车位宽度是否足够
    if config.parking.MIN_SLOT_WIDTH < config.vehicle.WIDTH + 0.3:
        return False, f"最小车位宽度({config.parking.MIN_SLOT_WIDTH}m)应大于车宽+安全间隙({config.vehicle.WIDTH + 0.3}m)"
    
    return True, "配置有效"


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 创建配置
    config = get_default_config()
    
    # 打印配置摘要
    config.print_summary()
    
    # 验证配置
    valid, msg = validate_config(config)
    print(f"\n配置验证: {msg}")
    
    # 获取场景特定配置
    config_narrow = get_config_for_scenario(3)
    print(f"\n狭窄场景 - 最小转弯半径: {config_narrow.vehicle.MIN_TURNING_RADIUS}m")