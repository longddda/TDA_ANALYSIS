# TDA Analysis Pipeline for Honeybee Group Dynamics

This repository contains a complete analysis pipeline for honeybee group experiments, starting from extracted video frames and ending with topological, change-point, and network-based analyses.

The workflow is:

1. Extract frames from the original video
2. Organize the frames in a folder
3. Generate representative point clouds from the images
4. Compute topological features and CROCKER plots
5. Detect temporal change points
6. Perform network-level analysis


## Overview

The main goal of this project is to convert experimental image sequences into structured quantitative descriptors of collective behavior.

More specifically, the pipeline is designed to:

- convert bee images into representative point clouds
- describe spatial organization using topological data analysis
- visualize topological evolution across time and scale with CROCKER plots
- identify transition times using change-point detection
- complement TDA results with graph-based network metrics

In short, the pipeline follows:

**image -> point cloud -> topology -> change point / network**


## Repository Structure

```text
TDA_code/
|-- 1bee_point_clouds.py
|-- 2crocker.py
|-- 3change_point.py
|-- 4network_analysis.py
`-- README.md
```

Script summary:

- `1bee_point_clouds.py`: image processing and point-cloud generation
- `2crocker.py`: Betti-0 computation and CROCKER plot generation
- `3change_point.py`: change-point detection on TDA-derived signals
- `4network_analysis.py`: epsilon-graph construction and network summary analysis


## Recommended Input Organization

The code assumes that video frames have already been extracted before analysis.

A recommended naming pattern is:

```text
frame_1.png
frame_2.png
frame_3.png
...
```

This naming scheme allows `1bee_point_clouds.py` to sort frames correctly by frame index.

A recommended directory structure is:

```text
frames/
    sample1/
        frame_1.png
        frame_2.png
        ...
    sample2/
        frame_1.png
        frame_2.png
        ...
```

If needed, frames can be extracted with `ffmpeg`, for example:

```bash
ffmpeg -i input.mp4 frames/frame_%d.png
```


## Quick Start

Run the pipeline in the following order:

1. Extract video frames and place them in a folder
2. Run `1bee_point_clouds.py` to generate a point-cloud `.npy` file
3. Run `2crocker.py` to compute `Betti_0` and generate the CROCKER plot
4. Run `3change_point.py` to detect change points
5. Run `4network_analysis.py` to compute network metrics and summary figures

The scripts currently use hard-coded paths and parameters in several places, so check them before execution.


## Pipeline Details

### 1. Point-Cloud Generation

Script: `1bee_point_clouds.py`

This script:

- reads frame images
- detects the dish region
- segments bee foreground regions
- extracts a fixed number of representative points from each frame
- saves the result as a point-cloud sequence

Key functions:

- `extract_bee_point_cloud(image_path, n_bees=19, debug=False)`
- `extract_point_cloud_for_folder(frame_dir, pattern="frame_*.png", n_bees=20, debug_one=False)`

Output format:

```python
(num_frames, n_bees, 2)
```

where:

- `num_frames` is the number of frames
- `n_bees` is the number of representative points per frame
- `2` corresponds to 2D coordinates `[x, y]`


### 2. TDA and CROCKER Plot

Script: `2crocker.py`

This script:

- loads the point-cloud sequence
- constructs a Rips complex for each frame
- computes `Betti_0(epsilon)`
- builds the full `Betti_0` matrix across time
- plots `Betti_0` at a fixed epsilon
- generates the CROCKER plot

Core idea:

- build a `RipsComplex` from each point cloud
- extract 0-dimensional persistence intervals
- count how many intervals are alive at each epsilon
- interpret this count as `Betti_0(epsilon)`

Main outputs:

- `betti0_matrix.npy`
- `epsilon_values.npy`
- `betti0_over_time_eps*.png`
- `crocker_betti0.png`


### 3. Change-Point Detection

Script: `3change_point.py`

This script:

- reads `betti0_matrix.npy`
- selects the `Betti_0` vector over a chosen epsilon range
- computes the L2 norm `||b(t)||_2`
- smooths the resulting time series
- applies `ruptures` for global change-point detection

Current method:

- compress each frame-wise `Betti_0` vector into one scalar
- run `Binseg(model="l2")`

Main outputs:

- `norm_series.npy`
- `norm_series_smooth.npy`
- `change_points_global.npy`
- `norm_cp_global.png`


### 4. Network Analysis

Script: `4network_analysis.py`

This script:

- builds an epsilon graph for each frame
- computes frame-wise network metrics
- extracts trial-level summary metrics
- generates boxplots and example time-series figures

Current network metrics:

- `gcc_ratio`: ratio of nodes in the largest connected component
- `edge_density`: undirected edge density

Trial-level summary metrics:

- `gcc_peak`
- `density_peak`
- `gcc_high_duration`
- `density_high_duration`
- `gcc_auc_norm`
- `density_auc_norm`


## Data Flow

```text
Original video
   -> frame extraction
frame_*.png
   -> 1bee_point_clouds.py
bee_point_clouds_*.npy
   -> 2crocker.py
betti0_matrix.npy + epsilon_values.npy + CROCKER plot
   -> 3change_point.py
change-point detection results

bee_point_clouds_*.npy
   -> 4network_analysis.py
network time series + summary figures
```


## Dependencies

The main Python packages used in this project are:

- `numpy`
- `scipy`
- `matplotlib`
- `opencv-python`
- `scikit-learn`
- `gudhi`
- `ruptures`
- `networkx`
- `pandas`
- `tqdm`

Installation:

```bash
pip install numpy scipy matplotlib opencv-python scikit-learn gudhi ruptures networkx pandas tqdm
```


## Parameters to Check Before Running

Several paths and analysis parameters are currently specified directly inside the scripts. Before running the pipeline, check:

- input image path / point-cloud path / output path
- `n_bees`
- `epsilon`
- `eps_max`
- `eps_step`
- `eps_target`
- smoothing window size
- number of change points `n_bkps`

In particular:

- input/output paths in `1bee_point_clouds.py`
- `npy_path` and output directory in `2crocker.py`
- input directory and change-point parameters in `3change_point.py`
- `data_root`, `out_root`, and `epsilon` in `4network_analysis.py`


## Notes

- The current TDA step focuses on `Betti_0`
- The current change-point analysis is applied to TDA-derived signals, not raw images
- The current network analysis is based on distance-thresholded epsilon graphs
- The scripts are designed as a sequential pipeline rather than as a packaged library


## Expected Outputs

Depending on which scripts are run, the pipeline produces:

- point-cloud `.npy` files
- `Betti_0` matrices
- CROCKER plots
- change-point detection outputs
- trial-level network metric tables
- summary figures for group comparison
