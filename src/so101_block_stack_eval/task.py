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
class _AtEnd:
    """An env flag on the trial's last step. A latched flag says whether it ever held."""

    name: str

    def __call__(self, record, target):
        steps = record.steps
        return Score(value=float(bool(steps and steps[-1].result.info.get(self.name))))


def stack(
    layouts: str | None = None,
    episodes: int | None = None,
    max_steps: int = 600,
    until_done: bool = False,
) -> Task:
    """One scene per line of ``layouts``, the first ``episodes`` of them.

    Each scene carries its block spawns, its colour pair and the arm's start, which the
    embodiment places. A trial succeeds the first step the stack holds, and ``lifted`` says
    whether the jaws ever held the named block off the table.

    With ``until_done``, for an embodiment made with it, the task is
    ``so101-block-stack-until-done`` and the trial runs until the policy stops it or
    ``max_steps`` runs out. The last step clears the arm away, and ``standing`` scores
    whether the stack is still seated without it. ``stacked`` says whether the stack
    ever held as the default trial scores it.
    """
    path = Path(layouts) if layouts else LAYOUTS
    with path.open() as file:
        rows = [json.loads(line) for line in file][:episodes]
    return Task(
        name="so101-block-stack-until-done" if until_done else "so101-block-stack",
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
        scorer=(
            [_AtEnd("standing"), _AtEnd("stacked"), _AtEnd("lifted")]
            if until_done
            else [success_at_end(), _AtEnd("lifted")]
        ),
        max_steps=max_steps,
    )
