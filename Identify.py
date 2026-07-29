import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# Configure matplotlib for better display
def setup_plot_style():
    """
    Configure matplotlib plot style
    """
    # Use default sans-serif font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10


# Initialize plot style
setup_plot_style()


def find_concave_points(vertices):
    """
    Find concave points of polygon (interior angle > 180 degrees)
    """
    concave = []
    n = len(vertices)
    for i in range(n):
        p1 = np.array(vertices[(i-1) % n])
        p2 = np.array(vertices[i])
        p3 = np.array(vertices[(i+1) % n])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate cross product (in OpenCV coordinate system)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # For clockwise polygon, positive cross product means concave point
        if cross > 0:
            concave.append(i)
    
    return concave


def remove_border(gray, border_threshold=50):
    """
    Detect and remove white border from image
    
    Parameters:
        gray: Grayscale image
        border_threshold: Pixel intensity threshold for white detection (default: 50)
                           Pixels with intensity > (255 - border_threshold) are considered white
                           Lower = more sensitive to borders
    
    Returns:
        Cropped image, border information
    """
    h, w = gray.shape
    
    # 计算白色阈值：像素值大于 (255 - border_threshold) 被认为是白色
    # 默认 border_threshold=50，所以白色阈值是 205
    white_threshold = 255 - border_threshold
    
    # 固定比例阈值 0.7（70% 以上的像素是白色才认为是边框）
    ratio_threshold = 0.7
    
    # Check if a region is mostly white
    def is_border_white(img, x, y, w, h):
        region = img[y:y+h, x:x+w]
        white_pixels = np.sum(region > white_threshold)
        total_pixels = region.size
        return white_pixels / total_pixels > ratio_threshold
    
    # Detect borders from outside to inside
    border_top = 0
    border_bottom = 0
    border_left = 0
    border_right = 0
    
    # Detect top border
    for i in range(min(h//3, 50)):
        if is_border_white(gray, 0, i, w, 1):
            border_top = i + 1
        else:
            break
    
    # Detect bottom border
    for i in range(min(h//3, 50)):
        if is_border_white(gray, 0, h-1-i, w, 1):
            border_bottom = i + 1
        else:
            break
    
    # Detect left border
    for i in range(min(w//3, 50)):
        if is_border_white(gray, i, border_top, 1, h - border_top - border_bottom):
            border_left = i + 1
        else:
            break
    
    # Detect right border
    for i in range(min(w//3, 50)):
        if is_border_white(gray, w-1-i, border_top, 1, h - border_top - border_bottom):
            border_right = i + 1
        else:
            break
    
    # Crop image to remove border
    cropped = gray[border_top:h-border_bottom, border_left:w-border_right]
    
    # Return cropped image and border info
    border_info = {
        'top': border_top,
        'bottom': border_bottom,
        'left': border_left,
        'right': border_right,
        'original_shape': (h, w),
        'cropped_shape': cropped.shape
    }
    
    return cropped, border_info


def merge_collinear_points(vertices, line_fit_threshold=2.0):
    """
    Merge collinear points on the same edge while preserving corners
    
    Parameters:
        vertices: List of vertices in order
        line_fit_threshold: Maximum distance from point to fitted line (in pixels)
    
    Returns:
        Merged vertices list
    """
    if len(vertices) <= 3:
        return vertices
    
    n = len(vertices)
    merged = []
    i = 0
    processed_count = 0
    
    # 防止死循环：最多处理 n 次迭代
    while i < n and processed_count < n:
        # 添加当前点
        merged.append(vertices[i])
        
        # 如果这是最后一个点，结束
        if i == n - 1:
            break
        
        # 查找从 i 开始的直线的终点
        # 初始化终点为下一个点
        end_idx = i + 1
        
        # 尝试扩展直线
        for candidate_end in range(i + 2, n):
            # 检查从 i 到 candidate_end 的所有点是否在一条直线上
            p1 = np.array(vertices[i], dtype=np.float32)
            p_end = np.array(vertices[candidate_end], dtype=np.float32)
            v = p_end - p1
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-6:
                # 起点和终点太近，无法定义直线
                end_idx = candidate_end
                continue
            
            # 检查中间所有点
            all_collinear = True
            for k in range(i + 1, candidate_end):
                p_k = np.array(vertices[k], dtype=np.float32)
                w = p_k - p1
                cross = np.cross(v, w)
                dist = abs(cross) / v_norm
                
                if dist > line_fit_threshold:
                    all_collinear = False
                    break
            
            if all_collinear:
                end_idx = candidate_end
            else:
                # 如果遇到不在直线上的点，停止扩展
                break
        
        # 记录处理了多少个点
        processed_count += (end_idx - i)
        
        # 移动到下一个未处理的点
        i = end_idx
        
        # 如果 i 没有前进，强制前进一位
        if i <= i:  # 这个条件永远为真，但为了安全保留
            pass
    
    # 如果 merge 后点数少于3，返回原始数据
    if len(merged) < 3:
        return vertices
    
    # 确保顺时针顺序
    merged = ensure_clockwise_order(merged)
    
    return merged


def ensure_clockwise_order(vertices):
    """
    Ensure polygon vertices are in clockwise order while preserving edge order
    """
    if len(vertices) <= 3:
        return vertices
    
    # Calculate signed area (shoelace formula)
    area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    
    # If area is positive (counter-clockwise), reverse order to make clockwise
    if area > 0:
        vertices = list(reversed(vertices))
    
    return vertices

def detect_polygons(image_path, epsilon_factor=0.003, min_area=100, max_vertices=50, 
                   auto_remove_border=True, merge_collinear=True, collinear_threshold=5.0,
                   visualize=True):
    """
    Detect white polygons on black background, auto-remove white border, 
    return vertex coordinates in edge order and visualize results
    
    Parameters:
        image_path: Image file path
        epsilon_factor: Polygon approximation accuracy (smaller = more accurate, more vertices)
        min_area: Minimum polygon area to filter noise
        max_vertices: Maximum number of vertices
        auto_remove_border: Whether to auto-remove white border
        merge_collinear: Whether to merge collinear points on straight edges
        collinear_threshold: Distance threshold for collinearity detection (pixels)
        visualize: Whether to display visualization (default: True)
    
    Returns:
        polygons: List of polygons, each is a list of vertices in edge order
    """
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return []
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    original_gray = gray.copy()
    
    # Auto-remove white border
    border_info = None
    if auto_remove_border:
        gray, border_info = remove_border(gray)
        if visualize:
            print(f"Detected and removed white border: top={border_info['top']}, bottom={border_info['bottom']}, left={border_info['left']}, right={border_info['right']}")
    
    # Gaussian blur to reduce noise (but preserve edges)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # For black background with white objects, use THRESH_BINARY
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    polygons = []
    
    # Create visualization image (using original image)
    vis_img = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    
    # Colors for different polygons
    colors = [
        (0, 0, 255),     # Red
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Cyan
        (128, 0, 255),   # Purple
        (0, 128, 255),   # Orange
    ]
    
    # Filter out border-like contours
    h, w = gray.shape
    
    # Use a separate counter for valid polygons
    polygon_counter = 0
    
    for idx, contour in enumerate(contours):
        # Calculate contour area, filter too small regions
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Filter out border-like contours
        is_border = False
        
        # Check if contour occupies most of the image
        if area > 0.8 * h * w:
            is_border = True
        
        # Check if contour is near image edge
        if (x < 5 or y < 5 or x + cw > w - 5 or y + ch > h - 5):
            if area > 0.5 * h * w:
                is_border = True
        
        if is_border:
            if visualize:
                print(f"Skipping possible border contour, area: {area:.0f}")
            continue
        
        # Polygon approximation using epsilon_factor
        peri = cv2.arcLength(contour, True)
        
        # Use epsilon_factor directly for approximation
        # But also try nearby values to find the best balance
        eps_values = [
            epsilon_factor * 0.3,   # More precise
            epsilon_factor * 0.5,   
            epsilon_factor * 0.7,   
            epsilon_factor,         # User-specified value
            epsilon_factor * 1.5,   
            epsilon_factor * 2.0,   # Less precise
            epsilon_factor * 3.0,
        ]
        
        best_approx = None
        best_vertices_count = 0
        
        for eps in eps_values:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            vertices_count = len(approx)
            
            # Prefer approximations with more vertices (more detail)
            # but respect the max_vertices limit
            if 3 <= vertices_count <= max_vertices:
                if vertices_count > best_vertices_count:
                    best_vertices_count = vertices_count
                    best_approx = approx
        
        # If no approximation found in the tested range, use epsilon_factor directly
        if best_approx is None:
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
            # If even that gives too many vertices, use a larger value
            if len(approx) > max_vertices:
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        else:
            approx = best_approx
        
        if len(approx) < 3:
            continue
        
        # Extract vertex coordinates
        vertices = approx.reshape(-1, 2).tolist()
        
        # Merge collinear points on straight edges
        if merge_collinear:
            original_count = len(vertices)
            vertices = merge_collinear_points(vertices, collinear_threshold)
            if len(vertices) < 3:
                # If merging resulted in less than 3 vertices, use original
                vertices = approx.reshape(-1, 2).tolist()
            else:
                reduced_count = original_count - len(vertices)
                if reduced_count > 0 and visualize:
                    print(f"  Polygon {polygon_counter + 1}: merged {reduced_count} collinear points ({original_count} -> {len(vertices)} vertices)")
        
        # Adjust coordinates back to original image position if border was removed
        if border_info is not None:
            vertices = [[x + border_info['left'], y + border_info['top']] for x, y in vertices]
        
        # Ensure clockwise order
        vertices_sorted = ensure_clockwise_order(vertices)
        
        # Save polygon information
        polygons.append(vertices_sorted)
        
        # Draw on visualization image with sequential numbering
        color = colors[polygon_counter % len(colors)]
        draw_polygon(vis_img, vertices_sorted, area, color, polygon_counter)
        
        # Increment the counter for valid polygons
        polygon_counter += 1
    
    # Display results only if visualize is True
    if visualize:
        visualize_results(original_gray, gray, binary, vis_img, polygons, border_info)
    else:
        # Print polygon information without visualization
        print_results(polygons)
    
    return polygons


def ensure_clockwise_order(vertices):
    """
    Ensure polygon vertices are in clockwise order while preserving edge order
    """
    if len(vertices) <= 3:
        return vertices
    
    # Calculate signed area (shoelace formula)
    area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    
    # If area is positive (counter-clockwise), reverse order to make clockwise
    if area > 0:
        vertices = list(reversed(vertices))
    
    return vertices


def draw_polygon(img, vertices, area, color, polygon_id):
    """
    Draw polygon on image
    
    Parameters:
        img: Image to draw on
        vertices: List of vertices
        area: Polygon area
        color: Color for drawing
        polygon_id: Sequential ID for the polygon (starting from 0)
    """
    pts = np.array([vertices], dtype=np.int32)
    
    # Draw filled polygon with transparency
    overlay = img.copy()
    cv2.fillPoly(overlay, pts, color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # Draw polygon outline
    cv2.polylines(img, pts, True, color, 3)
    
    # Draw and label vertices
    for i, (x, y) in enumerate(vertices):
        cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 8, color, 2)
        cv2.putText(img, f"v{i}", (x-10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show polygon ID (1-based for display) and vertex count
    x0, y0 = vertices[0]
    cv2.putText(img, f"P{polygon_id + 1}({len(vertices)}v)", (x0-30, y0-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def visualize_results(original, cropped, binary, processed, polygons, border_info=None):
    """
    Visualize results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image (with border)', fontsize=12)
    axes[0, 0].axis('off')
    if border_info:
        # Mark border area on original image
        h, w = original.shape
        rect = plt.Rectangle((border_info['left'], border_info['top']), 
                            w - border_info['left'] - border_info['right'],
                            h - border_info['top'] - border_info['bottom'],
                            linewidth=2, edgecolor='r', facecolor='none')
        axes[0, 0].add_patch(rect)
    
    # Image after border removal
    axes[0, 1].imshow(cropped, cmap='gray')
    axes[0, 1].set_title('After Border Removal', fontsize=12)
    axes[0, 1].axis('off')
    
    # Binary image
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary Image', fontsize=12)
    axes[0, 2].axis('off')
    
    # Detection results on original image
    result_on_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    
    for idx, vertices in enumerate(polygons):
        pts = np.array([vertices], dtype=np.int32)
        color = colors[idx % len(colors)]
        cv2.polylines(result_on_original, pts, True, color, 3)
        for i, (x, y) in enumerate(vertices):
            cv2.circle(result_on_original, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(result_on_original, str(i), (x-12, y-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    axes[1, 0].imshow(cv2.cvtColor(result_on_original, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Detection Results (Original)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Colored visualization
    axes[1, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Detailed Visualization', fontsize=12)
    axes[1, 1].axis('off')
    
    # Polygon information
    info_text = f"Detected {len(polygons)} polygons\n"
    for i, vertices in enumerate(polygons):
        info_text += f"P{i+1}: {len(vertices)} vertices\n"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Detection Info', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print polygon information
    print_results(polygons)


def print_results(polygons):
    """
    Print polygon detection results
    """
    print(f"\n{'='*60}")
    print(f"Detected {len(polygons)} white polygons")
    print(f"{'='*60}")
    for i, vertices in enumerate(polygons):
        print(f"\nPolygon {i+1}:")
        print(f"  Number of vertices: {len(vertices)}")
        print(f"  Vertex coordinates (edge order):")
        for j, (x, y) in enumerate(vertices):
            print(f"    {j}: ({x:>4}, {y:>4})")
        area = calculate_polygon_area(vertices)
        print(f"  Area: {area:.1f} pixels")
        print(f"  Direction: {'clockwise' if is_clockwise(vertices) else 'counter-clockwise'}")
        
        concave_points = find_concave_points(vertices)
        if concave_points:
            print(f"  Concave point indices: {concave_points}")
        else:
            print("  No concave points (convex polygon)")


def is_clockwise(vertices):
    """
    Check if polygon is clockwise
    """
    if len(vertices) < 3:
        return True
    
    area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    
    return area < 0


def calculate_polygon_area(vertices):
    """
    Calculate polygon area using shoelace formula
    """
    n = len(vertices)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# Example usage
if __name__ == "__main__":
    print("Current font configuration:")
    print(f"  matplotlib version: {matplotlib.__version__}")
    print(f"  Font list: {plt.rcParams['font.sans-serif'][:3]}")

    img_path = 'test_img/test02.png'

    # Run detection with merging
    polygons = detect_polygons(
        img_path, 
        epsilon_factor=0.003, 
        min_area=100, 
        auto_remove_border=True,
        merge_collinear=True,
        collinear_threshold=5.0,
        visualize=True
    )