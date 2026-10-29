# AI651 Assignment 2

This project contains the river forecasting proposal and the full two-seed Moving
MNIST experiment. The final learned model is D, wrapped in the eight-element
square-symmetry average. D has 480,784 parameters and one-step reach 12 original
pixels per coordinate, versus the baseline's 8.

## Run

From this directory, on the prepared host:

```bash
CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg ../../.venv/bin/python -u experiment.py > run.log 2>&1
```

The pipeline is resumable at epoch boundaries. It uses a lock to prevent duplicate
runs. An interrupted run restores Adam, sampling RNG and torch RNG along with
weights; cuDNN autotuning means bitwise cross-hardware reproducibility is not
promised. A completed experiment reuses its cached metrics. To rerun from scratch,
use a separate checkout/directory with empty `checkpoints/` and `artifacts/`.

For another host, create a Python 3.12 environment, install `requirements.txt`
with an appropriate CUDA-enabled torch build, and run `python experiment.py`.
The local `requirements-lock.txt` records the actual prepared environment,
including inherited packages unrelated to this assignment.

## Outputs

- `report/Final_Report.pdf` and `report/Final_Report.md`: final report.
- `report/figures/`: architecture diagram and Figures 1–6, including subpanels.
- `report/gifs/`: eight GIFs for Figures 3 and 6(a).
- `report/prediction_commits.txt`: required pre-experiment prediction commits.
- `report/completion_manifest.json`: completed experiment inventory.
- `artifacts/metrics_*.json`: unrounded results for both seeds.
- `artifacts/train_*.json`: all 30 epochs, losses and timings per learned model.
- `artifacts/figure_selection.json`: exact example video IDs and diagnostics.
- `checkpoints/`: final-epoch weights, Adam state and restart state.
- `status.json` and `run.log`: machine-readable progress and execution log.
- `Question1/`: original supplied companion notebook and bundled toy models.

Check progress with `cat status.json`; verify a completed run using
`../../.venv/bin/python verify.py`. Regenerate only the report with
`../../.venv/bin/python reporting.py`.

Every performance metric uses all 1,000 test videos (global indices 9000–9999).
Figure 1 uses only training videos 0–49. Table 1 uses synthetic tensors. The test
split never chooses an architecture. Predictions Q2.5 and Q2.11 are recorded and
committed automatically before their corresponding experiments. Only GPU 2 is
visible to the training process; no pre-existing process needed termination.

The code repository is local. No public GitHub repository or LMS submission has
been created; a public repository URL must be added to page one before submission
if required by the course. The report explicitly discloses AI assistance.
