#detect_cp_norm_global.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import ruptures as rpt  # pip install ruptures

def main():
    """
    全局在 0..T 上，对 ||b(t)||_2 做变点检测（Binseg, model='l2'）。
    不做按列减均值，只取 β0 向量的 L2 范数，再做轻微平滑。
    """

    # ================== 目录设置 ==================
    in_dir = "betti0_results_sample24"       # 之前算 β0 的输出目录
    out_dir = "norm_cp_global_sample24"
    os.makedirs(out_dir, exist_ok=True)

    # ================== 读取数据 ==================
    betti0_mat = np.load(os.path.join(in_dir, "betti0_matrix.npy"))   # (T, E)
    eps_values = np.load(os.path.join(in_dir, "epsilon_values.npy"))  # (E,)
    T, E = betti0_mat.shape
    print("betti0_mat shape =", betti0_mat.shape)

    # ================== 1. 构造 b(t) ==================
    # 只取 epsilon <= 250 的维度
    eps_max_use = 250.0
    idx_max = np.searchsorted(eps_values, eps_max_use)
    idx_max = max(1, min(idx_max, E))  # 防止越界

    # B[i, :] = 第 i 帧在选定 epsilon 范围内的 β0 向量
    B = betti0_mat[:, :idx_max].astype(float)  # (T, E_sub)

    # ================== 2. 直接求 L2 范数 ==================
    norm_series = np.linalg.norm(B, axis=1)  # shape (T,)

    # ================== 3. 轻微平滑 ==================
    smooth_window = 15          # 可以改 11 / 21 做鲁棒性检查
    norm_smooth = uniform_filter1d(norm_series, size=smooth_window)

    # ================== 4. 全局变点检测 ==================
    # 这里一次性在 [0, T) 上找 n_bkps 个变点
    n_bkps = 2   # 想只找一个就改成 1；想找 3 个就改成 3
    signal = norm_smooth.reshape(-1, 1)

    algo = rpt.Binseg(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)
    # ruptures 习惯：bkps 最后一个是 len(signal)，不是变点
    # 真正的变点帧索引列表：
    change_points = [cp for cp in bkps[:-1]]

    print(f"全局检测到的变点帧索引 = {change_points}")

    # ================== 5. 画图 ==================
    predator_frame = 120  # 实验记录的天敌加入帧

    x = np.arange(T)
    plt.figure(figsize=(10, 4))
    plt.plot(x, norm_series, color="lightgray", label="raw ||b(t)||")
    plt.plot(x, norm_smooth, color="C0", linewidth=2, label="smoothed ||b(t)||")

    # 实验天敌时间（黑线）
    plt.axvline(predator_frame, color="black", linewidth=2,
                label="Predator (exp.)")

    # 画出所有全局变点
    for i, cp in enumerate(change_points):
        plt.axvline(cp, color=f"C{1+i}", linestyle="--", linewidth=2,
                    label=f"CP{i+1} (global)")

    plt.xlim(0, T)
    plt.xlabel("Frame index")
    plt.ylabel(r"$\|\vec b(t)\|_2$")
    plt.title("L2 norm of β0-vector with global change points")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "norm_cp_global.png"), dpi=300)

    # ================== 6. 保存结果 ==================
    np.save(os.path.join(out_dir, "norm_series.npy"), norm_series)
    np.save(os.path.join(out_dir, "norm_series_smooth.npy"), norm_smooth)
    np.save(os.path.join(out_dir, "change_points_global.npy"),
            np.array(change_points, dtype=int))

if __name__ == "__main__":
    main()