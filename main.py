import cv2
import numpy as np
import time
import identify

try:
    img_path = 'test_img/001.png'
    img = cv2.imread(img_path)
except FileNotFoundError:
    print("Error: The specified image file was not found.")
    exit(0)

# 对黑色部分进行腐蚀
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()

# 2. 取反：黑色变白色，白色变黑色
inverted = cv2.bitwise_not(gray)

# 3. 对取反后的图像进行腐蚀（即对原图的黑色区域进行腐蚀）
kernel = np.ones((7, 7), np.uint8)
eroded_inverted = cv2.erode(inverted, kernel, iterations=1)

# 4. 再次取反，恢复原图色彩
gray = cv2.bitwise_not(eroded_inverted)

try:
    txt_path = 'start_end_pos.txt'
    file = open(txt_path, 'rb')
except IOError:
    print("Error: Could not open the text file.")
    exit(0)

# 窗口尺寸
WIDTH, HEIGHT = img.shape[1], img.shape[0]

# 从文本文件中读取起点和终点坐标，格式为两行，每行两个整数，分别表示起点和终点的 (x, y) 坐标
try:
    start_pos = tuple(map(int, file.readline().decode('utf-8').strip().split()))
    end_pos = tuple(map(int, file.readline().decode('utf-8').strip().split()))
except ValueError:
    print("Error: The text file does not contain valid coordinates.")
    file.close()
    exit(0)

if not (0 <= start_pos[0] < WIDTH and 0 <= start_pos[1] < HEIGHT):
    print("Error: Start position is out of image bounds.")
    file.close()
    exit(0)

if not (0 <= end_pos[0] < WIDTH and 0 <= end_pos[1] < HEIGHT):
    print("Error: End position is out of image bounds.")
    file.close()
    exit(0)

# Run detection with merging
polygons = identify.detect_polygons(
    gray, 
    epsilon_factor=0.003, 
    min_area=100, 
    auto_remove_border=True,
    merge_collinear=True,
    collinear_threshold=5.0,
    visualize=False
)

# 创建空白画布（背景为黑色）
canvas = img.copy()

# 动画循环
frame_count = 0
while True:
    # 每帧重新绘制背景（清空画布）
    canvas[:] = img
    # 显示起点和终点
    cv2.circle(canvas, start_pos, 5, (0, 255, 0), -1)  # 起点为绿色
    cv2.circle(canvas, end_pos, 5, (0, 0, 255), -1)    # 终点为红色

    # 绘制检测到的多边形
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.polylines(canvas, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # 显示窗口
    cv2.imshow('Animation Window', canvas)

    # 按 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cv2.destroyAllWindows()