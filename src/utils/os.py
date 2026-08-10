"""Local OS/path helpers not available in pinned mjlab builds."""

from __future__ import annotations

import re
from pathlib import Path


def get_latest_checkpoint_in_dir(
  run_dir: Path, checkpoint: str = "model_.*.pt"
) -> Path:
  """Return the latest checkpoint file directly inside a run directory."""
  if not run_dir.exists():
    raise ValueError(f"Run directory does not exist: {run_dir}")

  model_checkpoints = [
    f.name for f in run_dir.iterdir() if re.match(checkpoint, f.name)
  ]
  if len(model_checkpoints) == 0:
    raise ValueError(f"No checkpoint found in {run_dir} matching '{checkpoint}'")

  model_checkpoints.sort(key=lambda m: f"{m:0>15}")
  return run_dir / model_checkpoints[-1]
