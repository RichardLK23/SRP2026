"""
自动泊车路径规划主程序
"""
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from typing import List, Tuple

from config import get_default_config, validate_config
from svg_builder import SVGGuidePointExtractor
from hybrid_astar import HybridAStarPlanner
from bspline_smoother import BSplineSmoother
from visualization import Visualizer
from geometry import distance
from collision_checker import CollisionChecker  # 添加这个导入


def create_test_scenario(scenario_id: int = 1):
    """
    创建测试场景
    
    Args:
        scenario_id: 1-简单, 2-中等, 3-狭窄
    
    Returns:
        (obstacle_polys, start, goal)
    """
    if scenario_id == 1:
        # 简单场景：开阔停车场
        obstacles = [
            box(3, 3, 4, 4),
            box(7, 6, 8, 7),
            box(-2, 4, -1, 5),
        ]
        start = (0, 0, 0)
        goal = (5, 5, 0)
        
    elif scenario_id == 2:
        # 中等场景：带更多障碍物
        obstacles = [
            box(2, 0, 3, 8),
            box(5, 2, 6, 6),
            box(8, 1, 9, 7),
            box(1, -1, 9, 0),
        ]
        start = (1, 5, 0)
        goal = (7, 3, 0)
        
    else:
        # 狭窄场景：狭窄车位
        obstacles = [
            box(3, 2, 4, 4.5),
            box(3, 5.5, 4, 8),
            box(0, -1, 10, 0),
            box(0, 9, 10, 10),
        ]
        start = (1, 5, 0)
        goal = (4.5, 4, 0)
    
    # 确保所有障碍物都是有效的 Polygon 对象
    obstacle_polys = []
    for p in obstacles:
        try:
            if p.is_valid and not p.is_empty:
                obstacle_polys.append(p)
            else:
                # 尝试修复
                fixed = p.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    obstacle_polys.append(fixed)
        except Exception as e:
            print(f"创建障碍物警告: {e}")
            continue
    
    return obstacle_polys, start, goal


def main():
    """主函数"""
    print("=" * 60)
    print("自动泊车路径规划系统")
    print("=" * 60)
    
    # 1. 加载配置
    config = get_default_config()
    valid, msg = validate_config(config)
    if not valid:
        print(f"配置错误: {msg}")
        return
    print(f"配置验证: {msg}")
    
    # 2. 创建测试场景
    print("\n创建测试场景...")
    obstacle_polys, start, goal = create_test_scenario(1)
    print(f"有效障碍物数量: {len(obstacle_polys)}")
    print(f"起点: {start}")
    print(f"目标: {goal}")
    
    # 如果没有有效障碍物，创建默认障碍物
    if not obstacle_polys:
        print("警告：没有有效障碍物，创建默认边界")
        obstacle_polys = [box(-1, -1, 10, 10)]
    
    # 3. 模块1：提取引导点
    print("\n" + "=" * 60)
    print("模块1: 简化可视图引导点提取")
    print("=" * 60)
    
    try:
        guide_extractor = SVGGuidePointExtractor(config, obstacle_polys)
        guide_points = guide_extractor.extract_guide_points(start, goal)
        print(f"提取引导点 {len(guide_points)} 个:")
        for i, gp in enumerate(guide_points):
            print(f"  [{i}] ({gp[0]:.2f}, {gp[1]:.2f}, {math.degrees(gp[2]):.1f}°)")
    except Exception as e:
        print(f"引导点提取失败: {e}")
        guide_points = [start, goal]
    
    # 4. 模块2：混合A*搜索
    print("\n" + "=" * 60)
    print("模块2: 改进混合A*路径搜索")
    print("=" * 60)
    
    try:
        astar_planner = HybridAStarPlanner(config, obstacle_polys)
        rough_path = astar_planner.plan(start, goal, guide_points)
        
        if rough_path is None:
            print("混合A*搜索失败，使用直线路径")
            rough_path = [start, goal]
        else:
            print(f"生成粗糙路径，包含 {len(rough_path)} 个位姿点")
    except Exception as e:
        print(f"混合A*搜索异常: {e}")
        rough_path = [start, goal]
    
    # 5. 模块3：B样条平滑
    print("\n" + "=" * 60)
    print("模块3: B样条平滑与后处理")
    print("=" * 60)
    
    try:
        smoother = BSplineSmoother(config, obstacle_polys)
        smooth_path = smoother.smooth_path(rough_path)
        
        if smooth_path is None:
            print("路径平滑失败，使用粗糙路径")
            smooth_path = rough_path
        else:
            print(f"生成平滑路径，包含 {len(smooth_path)} 个位姿点")
    except Exception as e:
        print(f"路径平滑异常: {e}")
        smooth_path = rough_path
    
    # 6. 路径验证
    print("\n" + "=" * 60)
    print("路径验证")
    print("=" * 60)
    
    try:
        # 检查碰撞 - 直接使用已有的 collision_checker
        if 'astar_planner' in locals() and hasattr(astar_planner, 'collision_checker'):
            collision_checker = astar_planner.collision_checker
        else:
            # 创建新的 collision_checker
            collision_checker = CollisionChecker(obstacle_polys, config)
        
        has_collision = collision_checker.check_path_collision(smooth_path)
        print(f"碰撞检测: {'❌ 存在碰撞' if has_collision else '✅ 无碰撞'}")
    except Exception as e:
        print(f"碰撞检测异常: {e}")
    
    try:
        # 检查曲率
        if 'smoother' in locals():
            curvatures = smoother._calculate_curvatures(smooth_path)
            if curvatures:
                max_curv = max(abs(c) for c in curvatures)
                print(f"最大曲率: {max_curv:.4f} (限制: {config.bspline.MAX_CURVATURE:.4f})")
                print(f"曲率满足要求: {'✅ 是' if max_curv <= config.bspline.MAX_CURVATURE else '❌ 否'}")
    except Exception as e:
        print(f"曲率计算异常: {e}")
    
    # 路径长度
    try:
        path_length = 0
        for i in range(len(smooth_path) - 1):
            path_length += distance((smooth_path[i][0], smooth_path[i][1]),
                                   (smooth_path[i+1][0], smooth_path[i+1][1]))
        print(f"路径总长度: {path_length:.2f}m")
    except Exception as e:
        print(f"路径长度计算异常: {e}")
    
    # 7. 可视化
    print("\n生成可视化...")
    try:
        visualizer = Visualizer(config)
        visualizer.plot_scene(obstacle_polys, start, goal, guide_points, rough_path, smooth_path)
        plt.show()
    except Exception as e:
        print(f"可视化异常: {e}")
        # 简单可视化
        plt.figure(figsize=(10, 8))
        for poly in obstacle_polys:
            try:
                x, y = poly.exterior.xy
                plt.fill(x, y, 'gray', alpha=0.5)
            except:
                pass
        plt.plot(start[0], start[1], 'go', label='Start')
        plt.plot(goal[0], goal[1], 'ro', label='Goal')
        if smooth_path:
            sx = [p[0] for p in smooth_path]
            sy = [p[1] for p in smooth_path]
            plt.plot(sx, sy, 'b-', linewidth=2, label='Path')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    print("\n" + "=" * 60)
    print("规划完成！")
    print("=" * 60)
    
    return smooth_path


if __name__ == "__main__":
    main()