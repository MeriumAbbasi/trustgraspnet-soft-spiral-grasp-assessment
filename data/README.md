# Data layout

This repository is intentionally lightweight. Full raw trials, private robot XML files, and bulky processed tensors are **not** stored in Git.

## Recommended structure

- `data/demo/` — tiny demo subset committed to GitHub (a few `window_*.npz` files, `norm_stats.json`, `input_config.json`)
- `data/raw/` — full raw trial folders kept outside Git or downloaded separately
- `data/processed/` — full processed window dataset kept outside Git or downloaded separately

## Why this folder exists

The course repo requires:
1. code,
2. clear instructions,
3. a demo with sample inputs and outputs,
4. instructions for downloading data.

This `data/README.md` explains where each kind of file goes.

## What to put in `data/demo/`

A good demo subset is:
- 4-10 small `window_*.npz` files
- `norm_stats.json`
- `input_config.json`
- optionally `label_maps.json`
- optionally one example output JSON file produced by `scripts/demo_inference.py`

You can create this subset from a full processed dataset with:

```bash
python scripts/prepare_demo_subset.py   --source-data-dir /path/to/windowed_dataset   --outdir data/demo   --num-samples 8   --split test
```

## What `download_data.sh` is for

`download_data.sh` is only a convenience script for downloading **public** demo bundles, checkpoints, or processed subsets that you choose not to store in Git.
It is not meant for private robot XML files.

## Private / restricted assets

If the robot XML or hardware files cannot be shared, keep them out of the repo and document that clearly in `OMITTED_ASSETS.md`.
