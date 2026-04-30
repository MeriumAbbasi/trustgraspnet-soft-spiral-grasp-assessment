# Omitted assets

The following assets are intentionally **not** included in this repository:

1. Private MuJoCo / robot XML files used to instantiate the SpiRob simulator.
2. Full raw trial logs and bulky processed datasets.
3. Any lab-internal calibration files or hardware-specific connection details.

## Why they are omitted

These assets are excluded for size, privacy, or lab-IP reasons.

## What remains reproducible

The repository still contains:
- the model code,
- preprocessing code,
- training and evaluation scripts,
- a lightweight demo path based on precomputed windows,
- instructions for plugging in the private XML locally if authorized.

## How simulation-related scripts work

Scripts that require the private XML should be run locally with a user-supplied path, for example:

```bash
python scripts/collect_sim_dataset.py   --base-xml /absolute/path/to/private/spirob.xml   --outdir runs/raw_dataset
```

For the GitHub demo and grading, prefer the XML-free path based on `data/demo/` and `scripts/demo_inference.py`.
