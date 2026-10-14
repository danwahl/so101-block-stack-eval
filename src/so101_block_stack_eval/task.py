"""The block stack as an Inspect Robots task: one scene per held-out layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import Score, success_at_end
from inspect_robots.task import Task

from .spec import BLOCK_NAMES

LAYOUTS = Path(__file__).parent / "layouts.jsonl"


@dataclass(frozen=True)
class _Lifted:
    name: str = "lifted"

    def __call__(self, record, target):
        # The env latches it, so the last step says whether it ever happened.
        steps = record.steps
        return Score(value=float(bool(steps and steps[-1].result.info.get("lifted"))))


def stack(layouts: str | None = None, episodes: int | None = None, max_steps: int = 600) -> Task:
    """One scene per line of ``layouts``, the first ``episodes`` of them.

    Each scene carries its block spawns, its colour pair and the arm's start, which the
    embodiment places. A trial succeeds the first step the stack holds, and ``lifted`` says
    whether the jaws ever held the named block off the table.
    """
    path = Path(layouts) if layouts else LAYOUTS
    with path.open() as file:
        rows = [json.loads(line) for line in file][:episodes]
    return Task(
        name="so101-block-stack",
        scenes=[
            Scene(
                id=f"layout-{i:03d}",
                instruction=row["prompt"],
                target=Target(
                    "stack",
                    {"held": BLOCK_NAMES[row["held"]], "target": BLOCK_NAMES[row["target"]]},
                ),
                metadata={
                    "start": row["start"],
                    "positions": row["positions"],
                    "yaws": row["yaws"],
                    "held": BLOCK_NAMES[row["held"]],
                    "target": BLOCK_NAMES[row["target"]],
                },
            )
            for i, row in enumerate(rows)
        ],
        scorer=[success_at_end(), _Lifted()],
        max_steps=max_steps,
    )
