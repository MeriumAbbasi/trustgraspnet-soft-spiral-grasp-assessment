# Demo checklist

Before submitting, make sure the following command runs successfully on a clean machine:

```bash
python trustgraspnet_repo_ready/scripts/demo_inference.py \
  --checkpoint trustgraspnet_repo_ready/checkpoints/best.pt \
  --window trustgraspnet_repo_ready/data/demo/windows/window_0000123.npz \
  --stats trustgraspnet_repo_ready/data/demo/norm_stats.json \
  --input-config trustgraspnet_repo_ready/data/demo/input_config.json
```

A good demo should:
- finish in seconds,
- not require the private XML,
- print predictions for one sample input,
- optionally compare them with ground truth stored in the same `.npz` file.
