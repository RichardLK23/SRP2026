"""
Collision Detection Test - Independent Test File
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import math

from config import get_default_config
from collision_checker import CollisionChecker
from geometry import create_vehicle_rect, distance


def create_test_obstacles():
    """
    Create test obstacles
    """
    obstacles = [
        box(2.0, 2.0, 3.5, 3.5),
        box(6.0, 5.0, 7.5, 6.5),
        box(-1.0, 4.0, 0.5, 5.0),
    ]
    
    obstacle_polys = []
    for p in obstacles:
        try:
            if p.is_valid and not p.is_empty:
                obstacle_polys.append(p)
            else:
                fixed = p.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    obstacle_polys.append(fixed)
        except Exception as e:
            print(f"Obstacle warning: {e}")
            continue
    
    return obstacle_polys


def test_collision_detection():
    """
    Test collision detection
    """
    print("=" * 60)
    print("Collision Detection Test")
    print("=" * 60)
    
    config = get_default_config()
    print(f"Vehicle size: {config.vehicle.LENGTH}m x {config.vehicle.WIDTH}m")
    print(f"Clearance: {config.cost.OBSTACLE_CLEARANCE}m")
    
    obstacle_polys = create_test_obstacles()
    print(f"\nCreated {len(obstacle_polys)} obstacles:")
    for i, poly in enumerate(obstacle_polys):
        bounds = poly.bounds
        print(f"  Obstacle {i}: ({bounds[0]:.1f}, {bounds[1]:.1f}) - ({bounds[2]:.1f}, {bounds[3]:.1f})")
    
    collision_checker = CollisionChecker(obstacle_polys, config)
    print("\n" + "=" * 60)
    
    test_cases = [
        (0.0, 0.0, 0.0, False, "Start - should be safe"),
        (5.0, 5.0, 0.0, False, "Goal - should be safe"),
        (2.5, 2.5, 0.0, True, "Obstacle center - should collide"),
        (2.5, 3.0, 0.0, True, "Inside obstacle - should collide"),
        (1.5, 2.5, 0.0, False, "Near obstacle edge - should be safe"),
        (3.8, 3.0, 0.0, False, "Near obstacle - should be safe"),
        (6.8, 5.8, 0.0, True, "Second obstacle center - should collide"),
        (0.0, 4.5, 0.0, True, "Third obstacle center - should collide"),
        (1.0, 1.0, 0.0, False, "Bottom-left - should be safe"),
        (8.0, 8.0, 0.0, False, "Top-right - should be safe"),
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for x, y, theta, expected, desc in test_cases:
        is_collision = collision_checker.check_vehicle_collision(x, y, theta)
        
        status = "[PASS]" if is_collision == expected else "[FAIL]"
        result = "COLLIDE" if is_collision else "SAFE"
        expected_str = "COLLIDE" if expected else "SAFE"
        
        print(f"{status} Pos ({x:.1f}, {y:.1f}) -> {result} (Expected: {expected_str}) - {desc}")
        
        if is_collision == expected:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    
    print("\n" + "=" * 60)
    print("Path Collision Test:")
    print("-" * 60)
    
    path_through_obstacle = []
    for i in range(21):
        t = i / 20
        x = 0 + t * 5
        y = 0 + t * 5
        path_through_obstacle.append((x, y, 0.0))
    
    print("Path 1: From (0,0) to (5,5) - through obstacle")
    has_collision = collision_checker.check_path_collision(path_through_obstacle)
    print(f"  Collision: {'[COLLIDE]' if has_collision else '[SAFE]'}")
    
    path_around_obstacle = [
        (0.0, 0.0, 0.0),
        (1.0, 1.5, 0.0),
        (1.5, 3.0, 0.0),
        (1.5, 4.5, 0.0),
        (3.0, 5.0, 0.0),
        (5.0, 5.0, 0.0),
    ]
    smooth_path = []
    for i in range(len(path_around_obstacle) - 1):
        p1 = path_around_obstacle[i]
        p2 = path_around_obstacle[i + 1]
        dist = distance((p1[0], p1[1]), (p2[0], p2[1]))
        num_steps = max(5, int(dist / 0.1))
        for j in range(num_steps):
            t = j / num_steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            smooth_path.append((x, y, 0.0))
    smooth_path.append(path_around_obstacle[-1])
    
    print("\nPath 2: Around obstacle")
    has_collision = collision_checker.check_path_collision(smooth_path)
    print(f"  Collision: {'[COLLIDE]' if has_collision else '[SAFE]'}")
    
    print("\n" + "=" * 60)
    print("Generating visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Collision Detection Test')
    
    for idx, poly in enumerate(obstacle_polys):
        try:
            x, y = poly.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.7, label=f'Obstacle {idx}' if idx == 0 else '')
            ax.plot(x, y, 'k-', linewidth=2)
            center = poly.centroid
            ax.text(center.x, center.y, f'O{idx}', fontsize=8, ha='center', va='center')
        except Exception as e:
            print(f"Plot obstacle warning: {e}")
    
    for x, y, theta, expected, desc in test_cases:
        color = 'green' if not expected else 'red'
        marker = 'o' if not expected else 'x'
        size = 100 if not expected else 150
        ax.plot(x, y, marker=marker, color=color, markersize=size/10)
        vehicle_poly = create_vehicle_rect(x, y, theta, config.vehicle.LENGTH, config.vehicle.WIDTH)
        try:
            vx, vy = vehicle_poly.exterior.xy
            ax.plot(vx, vy, color=color, linewidth=1, alpha=0.5)
        except:
            pass
    
    path_x = [p[0] for p in path_through_obstacle]
    path_y = [p[1] for p in path_through_obstacle]
    ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Path through obstacle')
    
    smooth_x = [p[0] for p in smooth_path]
    smooth_y = [p[1] for p in smooth_path]
    ax.plot(smooth_x, smooth_y, 'g--', linewidth=2, alpha=0.7, label='Path around obstacle')
    
    ax.legend(loc='upper left')
    ax.set_xlim(-3, 10)
    ax.set_ylim(-2, 9)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTest complete!")


def test_vehicle_rect():
    """
    Test vehicle rectangle creation
    """
    print("\n" + "=" * 60)
    print("Vehicle Rectangle Test:")
    print("-" * 60)
    
    config = get_default_config()
    length = config.vehicle.LENGTH
    width = config.vehicle.WIDTH
    
    test_poses = [
        (0, 0, 0),
        (2, 2, math.pi/4),
        (5, 3, math.pi/2),
    ]
    
    for x, y, theta in test_poses:
        rect = create_vehicle_rect(x, y, theta, length, width)
        print(f"Pos ({x:.1f}, {y:.1f}, {math.degrees(theta):.0f} deg)")
        print(f"  Type: {type(rect)}")
        print(f"  Valid: {rect.is_valid if hasattr(rect, 'is_valid') else 'N/A'}")
        if hasattr(rect, 'bounds'):
            bounds = rect.bounds
            print(f"  Bounds: ({bounds[0]:.2f}, {bounds[1]:.2f}) - ({bounds[2]:.2f}, {bounds[3]:.2f})")
        print()


if __name__ == "__main__":
    test_vehicle_rect()
    test_collision_detection()