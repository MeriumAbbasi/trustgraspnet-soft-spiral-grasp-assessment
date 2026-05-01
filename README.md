# TrustGraspNet: IMU-Only Trust-Aware Grasp Assessment for Soft Spiral Robots

This repository contains the code for our course project on **IMU-only, trust-aware grasp assessment** for a soft spiral robot. The project studies whether distributed onboard IMUs, together with actuator-state signals, are sufficient for predicting grasp success, grasp quality, intervention actions, and sensor trust without relying on external cameras or tactile arrays.

## What is included

- `spirob/` core package
- `scripts/` training, evaluation, preprocessing, sensor-placement, export, and demo scripts
- `report/` final report PDF
- `data/README.md` with the expected dataset layout
- `download_data.sh` template for public demo/checkpoint downloads
- `OMITTED_ASSETS.md` documenting the private assets that are not shared

## What is intentionally not included

- private MuJoCo robot XML files
- full raw trial logs
- bulky processed datasets
- lab-internal hardware configuration files

See `OMITTED_ASSETS.md` for details.

## Demo

The recommended grading path is the **XML-free demo**:

1. Put a small processed subset under `data/demo/`.
2. Put one trained checkpoint under `checkpoints/best.pt`.
3. Run:

```bash
python scripts/demo_inference.py   --checkpoint checkpoints/best.pt   --window data/demo/windows/window_000123.npz   --stats data/demo/norm_stats.json   --input-config data/demo/input_config.json
```

This runs inference on one precomputed processed window and prints predicted:

grasp success
grasp quality
intervention action
object family
per-IMU trustPlace a tiny demo subset here.
This prints model predictions in JSON form and does not require the private robot XML.

## Minimal installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional packages for simulation, ONNX export, and hardware interfaces are listed in `requirements-optional.txt`.

## Example workflow

### A. Create raw simulation data (requires private XML)

```bash
python scripts/collect_sim_dataset.py   --base-xml /absolute/path/to/private/spirob.xml   --outdir runs/raw_dataset   --num-trials 1000
```

### B. Compute shape modes and select 8 IMUs

```bash
python scripts/compute_shape_modes.py   --raw-dir runs/raw_dataset   --outdir runs/shape_modes   --num-modes 6   --max-segments 24

python scripts/select_imu_locations.py   --modes-dir runs/shape_modes   --outdir runs/imu_placement   --num-sensors 8   --forbidden-indices 21,22,23   --min-spacing 2
```

### C. Build processed windows

If the private XML is available locally:

```bash
python scripts/build_windows.py   --raw-dir runs/raw_dataset   --base-xml /absolute/path/to/private/spirob.xml   --outdir runs/windowed_dataset   --imu-preset custom   --imu-indices 2,4,8,10,13,15,17,20   --actuator-preset disp_only
```

If the XML cannot be shared, the script can still infer the available IMU indices directly from the raw trial CSV files:

```bash
python scripts/build_windows.py   --raw-dir runs/raw_dataset   --outdir runs/windowed_dataset   --imu-preset custom   --imu-indices 2,4,8,10,13,15,17,20   --actuator-preset disp_only
```

### D. Train

```bash
python scripts/train.py   --data-dir runs/windowed_dataset   --outdir runs/student_baseline   --epochs 40
```

### E. Evaluate

```bash
python scripts/evaluate.py   --data-dir runs/windowed_dataset   --checkpoint runs/student_baseline/best.pt   --outdir runs/eval_student
```

## Repository structure

```text
spirob/                  core package
scripts/                 train / eval / preprocessing / demo
checkpoints/             place trained checkpoints here
configs/                 local private path placeholders
data/demo/               tiny committed demo subset
report/                  final report PDF
results/                 optional exported tables / figures
```

## Notes on restricted assets

The project contains simulation and hardware components. Some parts of that stack depend on private robot XML files that cannot be redistributed. That is documented explicitly in `OMITTED_ASSETS.md`. The demo path is designed so that graders can still run the ML portion without those files.
