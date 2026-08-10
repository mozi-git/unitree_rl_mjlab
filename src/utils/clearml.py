"""ClearML utilities used by the local training scripts."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def init_clearml_task(
  project_name: str,
  task_name: str,
  tags: Sequence[str] = (),
  auto_connect_frameworks: bool | Mapping[str, bool | dict] = True,
) -> str | None:
  """Initialize a ClearML task and return its task ID.

  Returns None when the clearml package is not installed.
  """
  try:
    from clearml import Task
  except ImportError:
    return None

  task = Task.init(
    project_name=project_name,
    task_name=task_name,
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks=auto_connect_frameworks,
  )
  if tags:
    task.add_tags(list(tags))
  return str(task.id)


def connect_clearml_configuration(
  section_name: str,
  configuration: Mapping[str, Any],
  task_id: str | None = None,
) -> None:
  """Attach structured configuration to a ClearML task."""
  try:
    from clearml import Task
  except ImportError:
    return

  task = Task.get_task(task_id=task_id) if task_id else Task.current_task()
  if task is None:
    return
  task.connect(dict(configuration), name=section_name)


def upload_clearml_artifact(
  artifact_name: str,
  local_path: Path,
  task_id: str | None = None,
) -> None:
  """Upload a local file as a ClearML artifact."""
  try:
    from clearml import Task
  except ImportError:
    return

  if not local_path.exists():
    return

  task = Task.get_task(task_id=task_id) if task_id else Task.current_task()
  if task is None:
    return
  task.upload_artifact(name=artifact_name, artifact_object=str(local_path))


def get_clearml_checkpoint_path(
  log_path: Path,
  task_id: str,
  checkpoint_name: str | None = None,
) -> tuple[Path, bool]:
  """Download a checkpoint from a ClearML task, with local caching."""
  try:
    from clearml import Task
  except ImportError as e:
    raise RuntimeError(
      "ClearML is not installed. Install the `clearml` package to use this feature."
    ) from e

  task = Task.get_task(task_id=task_id)
  artifacts = task.artifacts or {}
  checkpoint_candidates = [
    name for name in artifacts.keys() if re.match(r"^model_\d+\.pt$", name)
  ]

  if checkpoint_name is None:
    if not checkpoint_candidates:
      raise ValueError(f"No checkpoint artifacts found in ClearML task {task_id}.")
    checkpoint_file = max(
      checkpoint_candidates,
      key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
  else:
    if checkpoint_name not in artifacts:
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in ClearML task {task_id}. "
        f"Available: {sorted(artifacts.keys())}"
      )
    checkpoint_file = checkpoint_name

  download_dir = log_path / "clearml_checkpoints" / task_id
  checkpoint_path = download_dir / checkpoint_file
  was_cached = checkpoint_path.exists()

  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    local_copy = artifacts[checkpoint_file].get_local_copy()
    if local_copy is None:
      raise RuntimeError(
        f"Failed to download checkpoint '{checkpoint_file}' from task {task_id}."
      )
    shutil.copy2(local_copy, checkpoint_path)

  return checkpoint_path, was_cached
