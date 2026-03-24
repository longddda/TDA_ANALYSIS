
import os

# 固定并行线程数，防止 MKL + KMeans 的已知内存泄漏，并顺便让 warning 消失
os.environ["OMP_NUM_THREADS"] = "4"   # 控制OpenMP的线程数目
os.environ["MKL_NUM_THREADS"] = "4"   # 控制Inter MKL的数学库线程数目

from sklearn.cluster import KMeans    # k均值聚类算法
import cv2                            # OpenCV计算机视觉库
import numpy as np                    # 数值计算库
import matplotlib.pyplot as plt       # 数值可视化库
import glob                           # 文件路径模式匹配库



# =====================================================
# 1. 按“连通区域”分配点：每一坨蜜蜂至少 1 个点，剩下按面积分配
# =====================================================
def sample_points_by_components(bees_mask,
                                total_points=19,
                                min_area=80,
                                use_kmeans=True,
                                random_state=0):
    """
    根据蜜蜂区域的连通组件来采样点，确保每个连通区域至少有一个点
    
    参数：
    bees_mask : 2D uint8 二值图，非零像素 = 蜜蜂区域，0 = 背景
    total_points : 一帧中希望得到的代表点总数（你这里就是 30）
    min_area : 小于该面积的连通区域视为噪声，丢弃
    use_kmeans : True -> 区域内部用 KMeans 让点更均匀铺开
                 False -> 区域内部简单随机采样
    random_state : 随机数种子， 保证结果可重现
    
    返回：
    pts : 采样得到的点---->坐标数组，shape = (total_points, 2)
    """

    # 连通域分析：每一块连续的白色区域是一坨蜜蜂
    # 返回值：
    # num_labels: 连通区域总数（包括背景）
    # labels: 与输入图像同大小的矩阵，每个像素标记为对应的区域标号
    # stats: 每个区域的统计信息 [x, y, width, area]
    # centroids: 每个区域的中心点坐标
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (bees_mask > 0).astype(np.uint8), # 将二值化的图转为0-1格式
        connectivity=4                    # 4连通：只考虑上下左右四个方向
    )

    # 0 号是背景，从 1 开始才是真正的区域
    areas = stats[1:, cv2.CC_STAT_AREA]           # 提取所有区域的面积(num_labels-1,)
    labels_idx = np.arange(1, num_labels)         # 对应 label 编号 1..num_labels-1

    # 过滤掉太小的区域（噪声、小碎块）
    valid_mask = areas >= min_area
    valid_labels = labels_idx[valid_mask]         # 有效的区域标签
    valid_areas = areas[valid_mask]               # 有效的区域面积

    if len(valid_labels) == 0:
        raise RuntimeError("没有找到足够大的蜜蜂区域，请调小 min_area。")

    num_regions = len(valid_labels)    # 有效区域的数量

    # ------------- 情况 A：点数不够给每一坨都分 1 个（理论上不太会发生）-------------
    if total_points <= num_regions:
        # 只能选面积最大的 total_points 块，各给 1 个点（用重心）
        order = np.argsort(-valid_areas) # 按照面积大小降序排列
        chosen_labels = valid_labels[order[:total_points]]   #  选择前total_points个最大的区域

        all_pts = []
        for lab in chosen_labels:
            ys, xs = np.where(labels == lab)  # 获取该区域所有的像素坐标
            coords = np.stack([xs, ys], axis=1).astype(float)  # 组合成坐标数组
            center = coords.mean(axis=0)[None, :]              # 计算区域重心[x, y]
            all_pts.append(center)

        pts = np.vstack(all_pts)    #将所有的点垂直堆叠
        return pts

    # ------------- 情况 B：点数充足（正常）-------------
    # 先给每个区域 1 个点：保证“每一坨至少一点评”
    base_quota = np.ones(num_regions, dtype=int) # 每个连通区域基础配额为1
    remaining = total_points - base_quota.sum()  # 剩余可分配点数

    # 剩余点按面积比例分配（浮点配额）
    float_quota = remaining * (valid_areas / valid_areas.sum())  #按照面积比例计算浮点配额
    extra_quota = np.floor(float_quota).astype(int)              #向下取整得到整数配额
    rem_after_floor = remaining - extra_quota.sum()              #取整后剩余的点数

    # 按小数部分从大到小，再补上 rem_after_floor 个点
    frac = float_quota - extra_quota  # 小数部分
    order = np.argsort(-frac)         # 按照小数部分降序排列
    extra_quota[order[:rem_after_floor]] += 1   # 给小数部分最大的区域额外分配点数

    # 每个区域最终应该分配的点数
    per_region_points = base_quota + extra_quota
    assert per_region_points.sum() == total_points   # 确保总的点数正确

    all_pts = []  #存储所有采样点

    for lab, n_pts in zip(valid_labels, per_region_points):
        # 获取当前区域的所有像素位置坐标
        ys, xs = np.where(labels == lab)
        coords = np.stack([xs, ys], axis=1).astype(float)   # (Ni, 2)

        if coords.shape[0] == 0:  # 如果区域没有像素，就可以直接跳过
            continue

        if coords.shape[0] <= n_pts:
            # 区域像素数不够，允许重复采样
            idx = np.random.choice(coords.shape[0], size=n_pts, replace=True)
            sampled = coords[idx]
        else:
            if use_kmeans and n_pts > 1:
                # 区域内部再做一次 KMeans，让 n_pts 个点在这一坨里面尽量均匀铺开
                kmeans = KMeans(
                    n_clusters=n_pts,             # 聚类数量等于要采样的点数
                    random_state=random_state,    # 随机数种子
                    n_init=10                     # 用不同初始中心运行算法的次数
                )
                kmeans.fit(coords)                # 对区域内的像素坐标进行聚类
                sampled = kmeans.cluster_centers_ # 获取聚类中心作为采样点
            else:
                # 简单随机采样
                idx = np.random.choice(coords.shape[0], size=n_pts, replace=False)
                sampled = coords[idx]

        all_pts.append(sampled)

    pts = np.vstack(all_pts) #合并所有区域的点

    # 兜底：极端情况下有可能多/少 1–2 个点，简单裁剪/补齐
    if pts.shape[0] > total_points:
        pts = pts[:total_points] # 裁掉多余的点
    elif pts.shape[0] < total_points:
        # 从现有的点中随机选择补充
        extra_idx = np.random.choice(pts.shape[0],
                                     size=(total_points - pts.shape[0]),
                                     replace=True)
        pts = np.vstack([pts, pts[extra_idx]])

    return pts


# =====================================================
# 2. 单帧处理：检测培养皿 → 提取蜜蜂 mask → 生成 30 个点
#    加入“先 CLOSING 再 OPENING”的物理一点的形态学处理
#    + 最后一层 KMeans，保证同一只蜂不会有两个点
# =====================================================
def extract_bee_point_cloud(image_path, n_bees=19, debug=False):
    
    """
    从一张实验图片中提取 n_bees 个代表蜜蜂的点（点云）
    
    处理流程:
    1. 读取图像并转为灰度
    2. 检测培养皿圆形区域
    3. 创建圆形掩膜，只保留培养皿内部
    4. 使用Otsu阈值提取蜜蜂区域
    5. 形态学处理优化蜜蜂掩膜
    6. 按连通区域采样点
    7. 最终KMeans聚类确保点数准确
    
    参数:
    image_path : 输入图像路径
    n_bees : 要提取的点数
    debug : 是否显示调试图像
    
    返回:
    points : 蜜蜂点云坐标，shape = (n_bees, 2)，每行 [x, y]
    """

    # ---- 1. 读图 & 灰度 ----
    img_bgr = cv2.imread(image_path)  # 读取BGR图像
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  #将图片转为灰度图

    # ---- 2. 霍夫圆检测培养皿 ----
    gray_blur = cv2.medianBlur(gray, 5)  # 中值滤波去噪

    
    # 霍夫圆检测参数:
    # dp=1.2 : 累加器分辨率与图像分辨率的反比
    # minDist=500 : 检测到的圆之间的最小距离
    # param1=100 : Canny边缘检测的高阈值
    # param2=50 : 累加器阈值，值越小检测到的圆越多
    # minRadius=300, maxRadius=700 : 圆的半径范围
    
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,  # 检测方法
        dp=1.2,
        minDist=500,
        param1=100,
        param2=50,
        minRadius=300,
        maxRadius=700
    )

    if circles is None:
        raise RuntimeError("没有检测到培养皿圆，请调整 HoughCircles 参数。")

    #提取第一个检测的圆 （正常来说是培养皿）
    x, y, r = np.round(circles[0, 0]).astype(int) # 圆心坐标以及半径

    # ---- 3. 圆形掩膜：只保留培养皿内部 ----
    mask_dish = np.zeros_like(gray, dtype=np.uint8) # 创建全黑掩膜
    cv2.circle(mask_dish, (x, y), r, 255, thickness=-1) # 绘制白色圆形
    gray_dish = cv2.bitwise_and(gray, gray, mask=mask_dish) # 应用掩膜， 只保留圆形区域

    # ---- 4. Otsu 阈值：提取“比背景暗”的蜜蜂区域 ----
    # THRESH_BINARY_INV: 反向二值化（蜜蜂比背景暗）
    # THRESH_OTSU: 自动计算最佳阈值
    _, bees_mask = cv2.threshold(
        gray_dish,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ========= 物理一点的形态学处理 =========
    # 先 CLOSING：把同一只蜂身上的小裂缝、断块粘在一起
    # 闭运算：先膨胀后腐蚀，填充小洞和连接断开的区域
    kernel_close = np.ones((5, 5), np.uint8)
    bees_mask = cv2.morphologyEx(bees_mask, cv2.MORPH_CLOSE,
                                 kernel_close, iterations=1)

    # 再 OPENING：去掉孤立的小噪点
    # 开运算：先腐蚀后膨胀，去除小噪点
    kernel_open = np.ones((3, 3), np.uint8)
    bees_mask = cv2.morphologyEx(bees_mask, cv2.MORPH_OPEN,
                                 kernel_open, iterations=1)

    # 再强制限制在培养皿内部
    bees_mask = cv2.bitwise_and(bees_mask, bees_mask, mask=mask_dish)

    # ---- 5. 先按连通区域分配“较多”的点（比如 2*n_bees）----
    raw_points = sample_points_by_components(
        bees_mask,
        total_points=max(n_bees, 2 * n_bees),  # 比 30 多一些，让信息更丰富
        min_area=80,        # 这个阈值可以根据图像大小略微调一下
        use_kmeans=True,
        random_state=0
    )

    # ---- 6. 再用 KMeans 把这些 raw_points 压缩成恰好 n_bees 个中心 ----
    # 这样：同一只蜂如果还被分成几块 → raw_points 中在附近有多点
    # KMeans 会把它们聚成 1 个中心，避免你圈出来那种“一只蜂两个点”的情况
    if raw_points.shape[0] != n_bees:
        kmeans_final = KMeans(
            n_clusters=n_bees,
            random_state=0,
            n_init=10
        )
        kmeans_final.fit(raw_points)
        points = kmeans_final.cluster_centers_
    else:
        points = raw_points

    # ---- 7. 可视化检查（可选） ----
    if debug:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # (1) 原图 + 培养皿圆 + 最终点云
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.scatter(points[:, 0], points[:, 1],
                    s=60, c='yellow', edgecolors='black')
        circle = plt.Circle((x, y), r, color='cyan', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title("Point cloud overlaid on original image")
        plt.axis('off')
        plt.show()

        # (2) 蜜蜂 mask + 最终点云
        plt.figure(figsize=(8, 6))
        plt.imshow(bees_mask, cmap='gray')
        plt.scatter(points[:, 0], points[:, 1],
                    s=30, c='red')
        plt.title("Bee mask and final 30 points (after clustering)")
        plt.axis('off')
        plt.show()

    return points


# =====================================================
# 3. 批量处理一个文件夹：对所有 frame_*.png 生成点云序列
# =====================================================
def extract_point_cloud_for_folder(frame_dir,
                                   pattern="frame_*.png",
                                   n_bees=20,
                                   debug_one=False):
    """
    对某个文件夹里的所有帧生成点云序列
    返回：shape = (num_frames, n_bees, 2)
    """

    frame_paths = sorted(
        glob.glob(os.path.join(frame_dir, pattern)),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
    )

    all_frames_points = []

    for i, path in enumerate(frame_paths):
        print(f"Processing {i+1}/{len(frame_paths)}: {path}")
        dbg = debug_one and (i == 0)  # 只对第一帧打开 debug 看效果
        pts = extract_bee_point_cloud(path, n_bees=n_bees, debug=dbg)
        all_frames_points.append(pts)

    all_frames_points = np.stack(all_frames_points, axis=0)
    return all_frames_points


# =====================================================
# 4. 示例入口
# =====================================================
if __name__ == "__main__":
#     # 单帧测试：把路径换成你自己的某一张图
#     test_img = r"D:\longddda\frames\frame_40.png"
#     pts = extract_bee_point_cloud(test_img, n_bees=30, debug=True)
#     print("该帧点云前 5 个点：")
#     print(pts[:5])

    #如果要对整个文件夹做：
    pcs = extract_point_cloud_for_folder(r"D:\longddda\20\frame29",
                                         pattern="frame_*.png",
                                         n_bees=20,
                                         debug_one=True)
    np.save(r"D:\longddda\20\bee_cloud_20\bee_point_clouds_sample29.npy", pcs)