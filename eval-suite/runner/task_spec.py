"""Discover and load eval-suite tasks from eval-suite/tasks/<id>/task.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    id: str
    category: str
    summary: str
    task_dir: Path
    fixture_dir: Path
    prompt_path: Path
    checker_path: Path
    expects_report: bool
    fanout_required: bool
    timeout_s: int
    tags: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")


def load_task(task_dir: Path) -> TaskSpec:
    spec_path = task_dir / "task.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return TaskSpec(
        id=spec["id"],
        category=spec["category"],
        summary=spec["summary"],
        task_dir=task_dir,
        fixture_dir=task_dir / spec.get("fixture_dir", "fixture"),
        prompt_path=task_dir / spec.get("prompt_file", "prompt.md"),
        checker_path=task_dir / spec.get("checker", "checker.py"),
        expects_report=bool(spec.get("expects_report", False)),
        fanout_required=bool(spec.get("fanout_required", False)),
        timeout_s=int(spec.get("timeout_s", 300)),
        tags=tuple(spec.get("tags", [])),
    )


def discover_tasks(tasks_root: Path) -> list[TaskSpec]:
    tasks = []
    for task_dir in sorted(tasks_root.iterdir()):
        if (task_dir / "task.json").exists():
            tasks.append(load_task(task_dir))
    return tasks


def select_tasks(tasks: list[TaskSpec], selector: str) -> list[TaskSpec]:
    if selector == "all":
        return tasks
    wanted = {t.strip() for t in selector.split(",") if t.strip()}
    selected = [t for t in tasks if t.id in wanted]
    missing = wanted - {t.id for t in selected}
    if missing:
        raise SystemExit(f"unknown task id(s): {', '.join(sorted(missing))}")
    return selected
