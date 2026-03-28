import os
import glob

from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def sample_points_by_components(bees_mask, total_points=19, min_area=80, use_kmeans=True, random_state=0):
    areas = None
    labels = None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (bees_mask > 0).astype(np.uint8),
        connectivity=4
    )

    areas = stats[1:, cv2.CC_STAT_AREA]
    labels_idx = np.arange(1, num_labels)

    valid_mask = areas >= min_area
    valid_labels = labels_idx[valid_mask]
    valid_areas = areas[valid_mask]

    if len(valid_labels) == 0:
        raise RuntimeError("No valid bee regions were found. Try reducing min_area.")

    num_regions = len(valid_labels)

    if total_points <= num_regions:
        order = np.argsort(-valid_areas)
        chosen_labels = valid_labels[order[:total_points]]

        all_pts = []
        for lab in chosen_labels:
            ys, xs = np.where(labels == lab)
            coords = np.stack([xs, ys], axis=1).astype(float)
            center = coords.mean(axis=0)[None, :]
            all_pts.append(center)

        pts = np.vstack(all_pts)
        return pts

    base_quota = np.ones(num_regions, dtype=int)
    remaining = total_points - base_quota.sum()

    # Allocate extra samples in proportion to component area.
    float_quota = remaining * (valid_areas / valid_areas.sum())
    extra_quota = np.floor(float_quota).astype(int)
    rem_after_floor = remaining - extra_quota.sum()

    frac = float_quota - extra_quota
    order = np.argsort(-frac)
    extra_quota[order[:rem_after_floor]] += 1

    per_region_points = base_quota + extra_quota
    assert per_region_points.sum() == total_points

    all_pts = []

    for lab, n_pts in zip(valid_labels, per_region_points):
        ys, xs = np.where(labels == lab)
        coords = np.stack([xs, ys], axis=1).astype(float)

        if coords.shape[0] == 0:
            continue

        if coords.shape[0] <= n_pts:
            idx = np.random.choice(coords.shape[0], size=n_pts, replace=True)
            sampled = coords[idx]
        else:
            if use_kmeans and n_pts > 1:
                # Spread samples within a component instead of drawing clustered points.
                kmeans = KMeans(
                    n_clusters=n_pts,
                    random_state=random_state,
                    n_init=10
                )
                kmeans.fit(coords)
                sampled = kmeans.cluster_centers_
            else:
                idx = np.random.choice(coords.shape[0], size=n_pts, replace=False)
                sampled = coords[idx]

        all_pts.append(sampled)

    pts = np.vstack(all_pts)

    if pts.shape[0] > total_points:
        pts = pts[:total_points]
    elif pts.shape[0] < total_points:
        extra_idx = np.random.choice(pts.shape[0],
                                     size=(total_points - pts.shape[0]),
                                     replace=True)
        pts = np.vstack([pts, pts[extra_idx]])

    return pts


def extract_bee_point_cloud(image_path, n_bees=19, debug=False):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # Detect the dish first so segmentation is limited to the valid region.
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=500,
        param1=100,
        param2=50,
        minRadius=300,
        maxRadius=700
    )

    if circles is None:
        raise RuntimeError("Dish boundary was not detected. Adjust the HoughCircles parameters.")

    x, y, r = np.round(circles[0, 0]).astype(int)

    mask_dish = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_dish, (x, y), r, 255, thickness=-1)
    gray_dish = cv2.bitwise_and(gray, gray, mask=mask_dish)

    _, bees_mask = cv2.threshold(
        gray_dish,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Closing then opening helps connect fragmented bodies and suppress noise.
    kernel_close = np.ones((5, 5), np.uint8)
    bees_mask = cv2.morphologyEx(bees_mask, cv2.MORPH_CLOSE,
                                 kernel_close, iterations=1)

    kernel_open = np.ones((3, 3), np.uint8)
    bees_mask = cv2.morphologyEx(bees_mask, cv2.MORPH_OPEN,
                                 kernel_open, iterations=1)

    bees_mask = cv2.bitwise_and(bees_mask, bees_mask, mask=mask_dish)

    raw_points = sample_points_by_components(
        bees_mask,
        total_points=max(n_bees, 2 * n_bees),
        min_area=80,
        use_kmeans=True,
        random_state=0
    )

    # A final clustering step enforces exactly one representative center per bee.
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

    if debug:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.scatter(points[:, 0], points[:, 1],
                    s=60, c="yellow", edgecolors="black")
        circle = plt.Circle((x, y), r, color="cyan", fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title("Point cloud overlaid on original image")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(bees_mask, cmap="gray")
        plt.scatter(points[:, 0], points[:, 1],
                    s=30, c="red")
        plt.title("Bee mask and final 30 points (after clustering)")
        plt.axis("off")
        plt.show()

    return points


def extract_point_cloud_for_folder(frame_dir,
                                   pattern="frame_*.png",
                                   n_bees=20,
                                   debug_one=False):
    frame_paths = sorted(
        glob.glob(os.path.join(frame_dir, pattern)),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
    )

    all_frames_points = []

    for i, path in enumerate(frame_paths):
        print(f"Processing {i + 1}/{len(frame_paths)}: {path}")
        dbg = debug_one and (i == 0)
        pts = extract_bee_point_cloud(path, n_bees=n_bees, debug=dbg)
        all_frames_points.append(pts)

    all_frames_points = np.stack(all_frames_points, axis=0)
    return all_frames_points


if __name__ == "__main__":
#     test_img = r"D:\longddda\frames\frame_40.png"
#     pts = extract_bee_point_cloud(test_img, n_bees=30, debug=True)
#     print("First five points:")
#     print(pts[:5])

    pcs = extract_point_cloud_for_folder(r"D:\longddda\20\frame29",
                                         pattern="frame_*.png",
                                         n_bees=20,
                                         debug_one=True)
    np.save(r"D:\longddda\20\bee_cloud_20\bee_point_clouds_sample29.npy", pcs)
