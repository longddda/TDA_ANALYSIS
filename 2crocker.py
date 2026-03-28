# -*- coding: utf-8 -*-
"""
Compute Betti-0 curves and CROCKER plots from bee point clouds.
"""

import os

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def betti0_curve_for_point_cloud(point_cloud,
                                 eps_max=700.0,
                                 eps_step=1.0,
                                 max_dim=1):
    """
    Compute the Betti-0 curve for a single point cloud.
    """
    rips = gd.RipsComplex(points=point_cloud)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.persistence()

    eps_values = np.arange(0.0, eps_max + eps_step, eps_step)
    intervals0 = st.persistence_intervals_in_dimension(0)
    if intervals0.size == 0:
        return eps_values, np.zeros_like(eps_values, dtype=int)

    births = intervals0[:, 0]
    deaths = intervals0[:, 1]

    # Truncate infinite intervals so they can be counted on the epsilon grid.
    inf_cut = float(eps_max + eps_step)
    deaths = np.where(np.isinf(deaths), inf_cut, deaths)

    betti0 = np.zeros_like(eps_values, dtype=int)

    # Count intervals satisfying birth <= epsilon < death.
    for k, eps in enumerate(eps_values):
        betti0[k] = int(np.sum((births <= eps) & (eps < deaths)))

    return eps_values, betti0


def main():
    npy_path = "bee_point_clouds_sample24.npy"
    out_dir = "betti0_results_sample24"
    os.makedirs(out_dir, exist_ok=True)

    eps_max = 700.0
    eps_step = 1.0
    eps_target = 80.0
    crocker_eps_max = 250.0

    pcs = np.load(npy_path)
    if pcs.ndim != 3 or pcs.shape[2] not in (2, 3):
        raise ValueError(f"Unexpected point-cloud shape: {pcs.shape}. "
                         f"Expected (num_frames, N, 2) or (num_frames, N, 3).")

    num_frames, num_points, dim = pcs.shape
    print(f"Loaded point clouds: frames = {num_frames}, points/frame = {num_points}, dimension = {dim}")

    eps_values = np.arange(0.0, eps_max + eps_step, eps_step)
    n_eps = len(eps_values)
    betti0_mat = np.zeros((num_frames, n_eps), dtype=int)

    for i in tqdm(range(num_frames), desc="Computing Betti_0 curves"):
        pc = pcs[i]
        eps, b0 = betti0_curve_for_point_cloud(
            pc,
            eps_max=eps_max,
            eps_step=eps_step,
            max_dim=1
        )
        if i == 0:
            print(f"Frame 0: Betti_0 at epsilon = 0 is {b0[0]} (expected to be close to {num_points})")
        betti0_mat[i, :] = b0

    np.save(os.path.join(out_dir, "betti0_matrix.npy"), betti0_mat)
    np.save(os.path.join(out_dir, "epsilon_values.npy"), eps_values)

    idx_eps = int(round((eps_target - eps_values[0]) / eps_step))
    idx_eps = max(0, min(idx_eps, n_eps - 1))
    real_eps = eps_values[idx_eps]
    print(f"Target epsilon = {eps_target}, using epsilon = {real_eps}, index = {idx_eps}")

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

    idx_eps_max = int(np.floor((crocker_eps_max - eps_values[0]) / eps_step))
    idx_eps_max = max(1, min(idx_eps_max, n_eps - 1))

    mat = betti0_mat[:, :idx_eps_max + 1]
    eps_sub = eps_values[:idx_eps_max + 1]

    down_t = 20
    mat_down = mat[::down_t, :]
    time_down = np.arange(mat_down.shape[0]) * down_t

    X, Y = np.meshgrid(time_down, eps_sub)

    plt.figure(dpi=300, figsize=(10, 6))
    levels = 10
    plt.contour(X, Y, mat_down.T, levels=levels, colors="k", linewidths=0.6)
    hm = plt.contourf(X, Y, mat_down.T, levels=levels, cmap=plt.cm.jet_r)
    cbar = plt.colorbar(hm)
    cbar.set_label(r"$\beta_0$", rotation=0, labelpad=10)

    plt.xlabel("Frame index (downsampled)")
    plt.ylabel("Epsilon")
    plt.title(r"CROCKER plot of $\beta_0$ (time vs epsilon)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "crocker_betti0.png"), dpi=300)
    plt.close()

    print(f"All Betti_0 results were saved to: {out_dir}")


if __name__ == "__main__":
    main()
