"""Rescore ``lifted`` in eval logs by replaying each trial's logged actions.

    python scripts/rescore.py outputs/openai/gpt-6-luna/so101-block-stack_*.json

The sim is deterministic, so the replay reproduces each trial; a trial whose replay does
not end as logged, succeeding first on its last step or never, stops the rescore before
anything is written. Each log is rewritten in place, its first original kept beside it as
``.json.orig``.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from so101_block_stack_eval.embodiment import SimEmbodiment


def replay(embodiment, sample, actions_path):
    """The trial's final ``success`` and latched ``lifted`` flags, or None for success if
    the replay succeeded before its last step."""
    scene = Scene(
        id=sample["scene_id"], instruction=sample["instruction"], metadata=sample["scene_metadata"]
    )
    embodiment.reset(scene)
    with actions_path.open() as file:
        actions = [json.loads(line)["action"] for line in file if '"action"' in line]
    info = {"success": False, "lifted": False}
    for t, action in enumerate(actions):
        info = embodiment.step(Action(data=np.asarray(action, np.float64))).info
        if info["success"] and t < len(actions) - 1:
            return None, info["lifted"]
    return info["success"], info["lifted"]


def rescore(embodiment, path):
    log = json.loads(path.read_text())
    for sample in log["samples"]:
        for scores, trial in zip(sample["epochs"], sample["trial_metadata"], strict=True):
            if not scores:
                continue
            success, lifted = replay(embodiment, sample, path.parent / trial["actions"])
            if success is None or float(success) != scores["success_at_end"]:
                raise SystemExit(f"{path} {sample['scene_id']}: replay does not reproduce")
            scores["lifted"] = float(lifted)
        scored = [scores["lifted"] for scores in sample["epochs"] if scores]
        if scored:
            sample["reduced"]["lifted"] = sum(scored) / len(scored)
    reduced = [s["reduced"]["lifted"] for s in log["samples"] if "lifted" in s["reduced"]]
    log["results"]["metrics"]["lifted"] = sum(reduced) / len(reduced)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", type=Path, nargs="+", help="Inspect Robots eval logs")
    args = parser.parse_args()

    embodiment = SimEmbodiment()
    try:
        logs = [(path, rescore(embodiment, path)) for path in args.logs]
    finally:
        embodiment.close()
    for path, log in logs:
        original = path.with_suffix(".json.orig")
        if not original.exists():
            shutil.copy2(path, original)
        path.write_text(json.dumps(log, indent=2))
        print(f"{path}: lifted {log['results']['metrics']['lifted']:.2f}")


if __name__ == "__main__":
    main()
