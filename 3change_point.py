import os

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
from scipy.ndimage import uniform_filter1d


def main():
    """
    Detect global change points from the L2 norm of the Betti-0 vector.
    """
    in_dir = "betti0_results_sample24"
    out_dir = "norm_cp_global_sample24"
    os.makedirs(out_dir, exist_ok=True)

    betti0_mat = np.load(os.path.join(in_dir, "betti0_matrix.npy"))
    eps_values = np.load(os.path.join(in_dir, "epsilon_values.npy"))
    T, E = betti0_mat.shape
    print("betti0_mat shape =", betti0_mat.shape)

    eps_max_use = 250.0
    idx_max = np.searchsorted(eps_values, eps_max_use)
    idx_max = max(1, min(idx_max, E))
    B = betti0_mat[:, :idx_max].astype(float)

    norm_series = np.linalg.norm(B, axis=1)

    smooth_window = 15
    norm_smooth = uniform_filter1d(norm_series, size=smooth_window)

    n_bkps = 2
    signal = norm_smooth.reshape(-1, 1)

    algo = rpt.Binseg(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)

    # `ruptures` appends the signal end as the final boundary.
    change_points = [cp for cp in bkps[:-1]]

    print(f"Detected global change points: {change_points}")

    predator_frame = 120

    x = np.arange(T)
    plt.figure(figsize=(10, 4))
    plt.plot(x, norm_series, color="lightgray", label="raw ||b(t)||")
    plt.plot(x, norm_smooth, color="C0", linewidth=2, label="smoothed ||b(t)||")

    # Plot the experimental event and detected change points.
    plt.axvline(predator_frame, color="black", linewidth=2,
                label="Predator (exp.)")

    for i, cp in enumerate(change_points):
        plt.axvline(cp, color=f"C{1 + i}", linestyle="--", linewidth=2,
                    label=f"CP{i + 1} (global)")

    plt.xlim(0, T)
    plt.xlabel("Frame index")
    plt.ylabel(r"$\|\vec b(t)\|_2$")
    plt.title("L2 norm of the Betti-0 vector with global change points")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "norm_cp_global.png"), dpi=300)

    np.save(os.path.join(out_dir, "norm_series.npy"), norm_series)
    np.save(os.path.join(out_dir, "norm_series_smooth.npy"), norm_smooth)
    np.save(os.path.join(out_dir, "change_points_global.npy"),
            np.array(change_points, dtype=int))


if __name__ == "__main__":
    main()
