# Phase-Time Array Enabled Multistatic Sensing with Multi-Level Fusion for UAV Localization

## Related Publication

📄 **Paper**: [*Phase-Time Array Enabled Multistatic Sensing with Multi-Level Fusion for UAV Localization*](https://arxiv.org/abs/2605.04919)  
*Ming Gao, Jianhua Mo, Meixia Tao*, accepted to IEEE GLOBECOM 2026 Workshops.

This repository provides the MATLAB and Python implementation for the paper, which studies multistatic UAV localization with phase-time array (PTA) receivers and multi-level fusion.

---

## Introduction

A PTA receiver uses rainbow beamforming to map directions onto OFDM subcarriers, so it estimates both the bistatic range and the angle of arrival (AoA) with a single RF chain. Two such receivers observe a UAV illuminated by a wide-beam transmitter, and their estimates are fused into a 2D position.

This codebase implements:
1) **Simulation**: bistatic echo channel, the rainbow-beam PTA receiver, and fully digital (FD) and phased-array (PA) receivers for comparison.
2) **Parameter estimation**: AoA from the strongest subcarrier and bistatic range from a sub-band IFFT.
3) **Parameter-level fusion**: GDOP-WLS (GDOP-weighted initialization and sigma-normalized least squares) and PF-MLP, with the path-loss-weighted baselines GI-PL and GDOP-PL.
4) **Signal-level fusion**: SF-CNN, which regresses the position directly from the received PTA amplitudes.

---

## File Structure and Function Description

| Category / Method | Filename | Main Function Description |
|---|---|---|
| Simulation | `simulation/run_simulation.m` | Monte Carlo evaluation of all fusion schemes over the transmit power sweep and of the FD / PTA / PA receivers (Tables II-III, data of Figs. 3-5). |
|  | `simulation/plot_results.m` | Figs. 3-5 from a result file. |
|  | `simulation/calibrate_sigma.m` | Error standard deviation of the PTA range and AoA estimates per transmit power, used by GDOP-WLS. |
|  | `simulation/generate_training_data.m` | Training data for PF-MLP and SF-CNN. |
| Signal model and estimation | `simulation/lib/system_params.m` | System parameters (Table I) and calibrated error standard deviations. |
|  | `simulation/lib/simulate_element_signals.m`, `rx_observation.m` | Echo channel, element-level signal and noise, and the FD / PTA / PA receiver outputs. |
|  | `simulation/lib/estimate_params_pta.m`, `_fd.m`, `_pa.m` | Bistatic range and AoA estimation. |
| Proposed GDOP-WLS | `simulation/lib/fuse_gdop_wls.m`, `gdop_init.m` | GDOP-weighted initialization and sigma-normalized least squares. |
| Proposed PF-MLP / SF-CNN | `simulation/lib/fuse_pf_mlp.m`, `fuse_sf_cnn.m` | Network inference through ONNX Runtime. |
| Baselines | `simulation/lib/fuse_pl_weighted.m` | GI-PL and GDOP-PL. |
|  | `simulation/lib/fuse_pa_range_only.m` | Range-only least squares for the PA receiver. |
| Model training | `model_training/train.py`, `evaluate.py`, `export_to_onnx.py` | Training, test-set evaluation and ONNX export of PF-MLP and SF-CNN. |
|  | `model_training/configs/` | Training configurations of the two networks. |

---

## Quick Start

### 1. Environment Setup

- MATLAB with the Optimization Toolbox (tested with R2023a)
- [uv](https://docs.astral.sh/uv/) for the Python environment of the networks:

```bash
cd model_training
uv sync
```

MATLAB runs the ONNX models through this environment (Python 3.10, `python_exe` in `run_simulation.m`). For GPU inference in MATLAB (`nn_device = 'cuda:0'`), install `onnxruntime-gpu` instead of `onnxruntime`.

### 2. Simulation

```matlab
cd simulation
run_simulation
plot_results
```

`run_simulation` prints Tables II and III and the RMSE of every scheme versus the transmit power, and saves the estimates to `simulation/results/`. `plot_results` writes Figs. 3-5 to `simulation/figures/`. Without trained networks, PF-MLP and SF-CNN are skipped.

**Main Parameter Descriptions** (top of `run_simulation.m`):
- `num_trials`: Monte Carlo trials per transmit power (default 10,000)
- `ptx_list`: transmit powers in dBm (default 42:2:52)
- `rx_compare_ptx`: transmit power of the FD / PTA / PA comparison (default 52)
- `seed`: random seed
- `model_dir`, `nn_train_ptx`: folder of the ONNX models, and the power at which the generalized models are trained
- `nn_device`: `'cpu'` or `'cuda:<id>'`

### 3. Train PF-MLP and SF-CNN

1) Generate 100,000 training samples at the transmit power `ptx` set in the script:

```matlab
cd simulation
generate_training_data   % writes model_training/data/pfmlp_Ptx52dBm.mat and sfcnn_Ptx52dBm.mat
```

2) Train, evaluate on the test split, and export to ONNX:

```bash
cd model_training
uv run python train.py --config configs/pfmlp.yaml
uv run python evaluate.py --experiment_dir saved_models/pfmlp_Ptx52dBm
uv run python export_to_onnx.py --experiment_dir saved_models/pfmlp_Ptx52dBm
```

The same with `configs/sfcnn.yaml`. Any config entry can be overridden, e.g. `--override data.path=data/sfcnn_Ptx44dBm.mat --override output.experiment_name=sfcnn_Ptx44dBm`.

3) Copy `<name>.onnx` and `<name>.json` from `saved_models/<name>/` to `simulation/models/`.

The models trained at 52 dBm are used at every transmit power; the models trained at the other powers are the power-matched variants in Fig. 5.

### 4. Measurement Error Calibration

```matlab
cd simulation
calibrate_sigma
```

Estimates the standard deviation of the PTA range and AoA errors for each link and transmit power and writes `results/sigma_table.csv`. The values go into `p.sigma_d` and `p.sigma_theta_deg` in `lib/system_params.m`.

---

## Citation

```bibtex
@article{gao2026pta,
  title   = {Phase-Time Array Enabled Multistatic Sensing with Multi-Level Fusion for {UAV} Localization},
  author  = {Gao, Ming and Mo, Jianhua and Tao, Meixia},
  journal = {arXiv preprint arXiv:2605.04919},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE).
