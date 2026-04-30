This repository includes a lightweight XML-free demo under data/demo/ so the model can be evaluated without MuJoCo assets or private robot design files.

Run:

python scripts/demo_inference.py \
  --checkpoint checkpoints/best.pt \
  --window data/demo/windows/window_000123.npz \
  --stats data/demo/norm_stats.json \
  --input-config data/demo/input_config.json

This runs inference on one precomputed processed window and prints predicted:

grasp success
grasp quality
intervention action
object family
per-IMU trustPlace a tiny demo subset here.
