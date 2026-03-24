# -*- coding: utf-8 -*-
"""
从 bee_point_clouds.npy 读取点云 (num_frames, N, 2/3)
只计算 Betti_0(epsilon)，并画：
  1) 固定 epsilon 的 β0 随帧数变化
  2) β0 的 CROCKER 图 (time vs epsilon)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd


# ---------------------------------------------------------
# 1. 对单帧点云计算 β0(eps) —— 改为 intervals 计数法（更稳）
# ---------------------------------------------------------
def betti0_curve_for_point_cloud(point_cloud,
                                 eps_max=700.0,
                                 eps_step=1.0,
                                 max_dim=1):
    """
    对单个点云计算 Betti_0(epsilon) 曲线

    β0(ε) 定义为：在 filtration value <= ε 的复形中，连通分量个数
    等价实现：统计 0维持久区间 (birth, death) 中满足 birth<=ε<death 的数量

    参数
    ----
    point_cloud : ndarray, 形状 (N, d)
    eps_max     : epsilon 最大值
    eps_step    : epsilon 步长
    max_dim     : Rips 复形最大维度，只算 β0 用 1 就够

    返回
    ----
    eps_values  : 一维数组, 所有 epsilon
    betti0      : 一维数组, β0(epsilon)
    """
    rips = gd.RipsComplex(points=point_cloud)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.persistence()

    eps_values = np.arange(0.0, eps_max + eps_step, eps_step)

    # 取 0 维的持久区间 (birth, death)
    intervals0 = st.persistence_intervals_in_dimension(0)
    if intervals0.size == 0:
        # 极端情况：没算到区间（理论上几乎不会）
        return eps_values, np.zeros_like(eps_values, dtype=int)

    births = intervals0[:, 0]
    deaths = intervals0[:, 1]

    # death=inf 的区间代表“永生”的连通分量：截断到 eps_max+eps_step
    inf_cut = float(eps_max + eps_step)
    deaths = np.where(np.isinf(deaths), inf_cut, deaths)

    betti0 = np.zeros_like(eps_values, dtype=int)

    # 统计 birth<=ε<death 的数量
    for k, eps in enumerate(eps_values):
        betti0[k] = int(np.sum((births <= eps) & (eps < deaths)))

    return eps_values, betti0


# ---------------------------------------------------------
# 2. 主函数：读取 npy，批量算 β0
# ---------------------------------------------------------
def main():
    # ===== 参数部分（可以按需修改） =====
    npy_path = "bee_point_clouds_sample24.npy"  # 点云数据
    out_dir = "betti0_results_sample24"         # 输出目录
    os.makedirs(out_dir, exist_ok=True)

    eps_max = 700.0     # epsilon 最大值（要覆盖 Rips 半径范围）
    eps_step = 1.0      # epsilon 步长

    # 用于画 β0–时间 的那个 epsilon（你觉得 80 合适就用 80）
    eps_target = 80.0

    # CROCKER 图只显示的 epsilon 最大值（方便放大底部细节）
    crocker_eps_max = 250.0   # 如果想看全部就改回 700

    # ===== 1. 读取点云 =====
    pcs = np.load(npy_path)  # (num_frames, N, 2/3)
    if pcs.ndim != 3 or pcs.shape[2] not in (2, 3):
        raise ValueError(f"点云形状不符合预期: {pcs.shape}，"
                         f"应为 (num_frames, N, 2) 或 (num_frames, N, 3)")

    num_frames, num_points, dim = pcs.shape
    print(f"读取点云: 帧数 = {num_frames}, 每帧点数 = {num_points}, 维度 = {dim}")

    eps_values = np.arange(0.0, eps_max + eps_step, eps_step)
    n_eps = len(eps_values)

    betti0_mat = np.zeros((num_frames, n_eps), dtype=int)

    # ===== 2. 对每一帧计算 β0(eps) 曲线 =====
    for i in tqdm(range(num_frames), desc="Computing Betti_0 curves (intervals)"):
        pc = pcs[i]
        eps, b0 = betti0_curve_for_point_cloud(
            pc,
            eps_max=eps_max,
            eps_step=eps_step,
            max_dim=1
        )
        if i == 0:
            print(f"Frame 0: epsilon=0 时 β0 = {b0[0]} (应接近点数 {num_points})")
        betti0_mat[i, :] = b0

    # 保存数据，方便后处理
    np.save(os.path.join(out_dir, "betti0_matrix.npy"), betti0_mat)
    np.save(os.path.join(out_dir, "epsilon_values.npy"), eps_values)

    # -----------------------------------------------------
    # 3. 画：固定 epsilon 的 β0 vs 帧数
    # -----------------------------------------------------
    idx_eps = int(round((eps_target - eps_values[0]) / eps_step))
    idx_eps = max(0, min(idx_eps, n_eps - 1))
    real_eps = eps_values[idx_eps]
    print(f"固定 epsilon = {eps_target}，实际使用 = {real_eps}，索引 = {idx_eps}")

    beta0_over_time = betti0_mat[:, idx_eps]

    plt.figure(figsize=(10, 4))
    plt.plot(beta0_over_time, label=rf"$\beta_0(\epsilon={real_eps:.1f})$")
    plt.xlabel("Frame index")
    plt.ylabel(r"$\beta_0$")
    plt.title(f"Betti_0 over time (epsilon = {real_eps:.1f})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,
                             f"betti0_over_time_eps{real_eps:.1f}.png"),
                dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 4. 画 β0 的 CROCKER 图 (time vs epsilon)
    # -----------------------------------------------------
    idx_eps_max = int(np.floor((crocker_eps_max - eps_values[0]) / eps_step))
    idx_eps_max = max(1, min(idx_eps_max, n_eps - 1))

    mat = betti0_mat[:, :idx_eps_max + 1]          # (num_frames, E')
    eps_sub = eps_values[:idx_eps_max + 1]         # 对应的 epsilon

    down_t = 20   # 每 20 帧取一次
    mat_down = mat[::down_t, :]                    # (T', E')
    time_down = np.arange(mat_down.shape[0]) * down_t

    X, Y = np.meshgrid(time_down, eps_sub)

    plt.figure(dpi=300, figsize=(10, 6))
    levels = 10
    plt.contour(X, Y, mat_down.T, levels=levels, colors='k', linewidths=0.6)
    hm = plt.contourf(X, Y, mat_down.T, levels=levels, cmap=plt.cm.jet_r)
    cbar = plt.colorbar(hm)
    cbar.set_label(r"$\beta_0$", rotation=0, labelpad=10)

    plt.xlabel("Frame index (downsampled)")
    plt.ylabel("Epsilon")
    plt.title(r"CROCKER plot of $\beta_0$ (time vs epsilon)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "crocker_betti0.png"), dpi=300)
    plt.close()

    print(f"所有 β0 相关结果已保存到文件夹: {out_dir}")


if __name__ == "__main__":
    main()
