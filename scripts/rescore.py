"""Rescore ``lifted`` and ``standing`` by replaying each trial's logged actions.

    python scripts/rescore.py outputs/openai/gpt-6-luna/so101-block-stack_*.json

The logs are the default task's, whose trials end on the first step the stack holds.
Each replay stops on its last logged action as an until-done trial would, so
``standing`` is scored as that task scores it. The sim is deterministic, so the replay
reproduces each trial; a trial whose replay does not end as logged, succeeding first on
its last step or never, stops the rescore before anything is written. Each log is
rewritten in place, its first original kept beside it as ``.json.orig``.
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
    """Whether the trial first stacked on its last step, or None if it stacked before it,
    the latched ``lifted`` flag, and whether the stack stands once the arm is cleared."""
    scene = Scene(
        id=sample["scene_id"], instruction=sample["instruction"], metadata=sample["scene_metadata"]
    )
    embodiment.reset(scene)
    with actions_path.open() as file:
        actions = [json.loads(line)["action"] for line in file if '"action"' in line]
    info = {"stacked": False, "lifted": False, "standing": False}
    for t, action in enumerate(actions):
        last = t == len(actions) - 1
        meta = {"request_stop": True} if last else {}
        info = embodiment.step(Action(data=np.asarray(action, np.float64), meta=meta)).info
        if info["stacked"] and not last:
            return None, info["lifted"], False
    return info["stacked"], info["lifted"], info["standing"]


def rescore(embodiment, path):
    log = json.loads(path.read_text())
    for sample in log["samples"]:
        for scores, trial in zip(sample["epochs"], sample["trial_metadata"], strict=True):
            if not scores:
                continue
            success, lifted, standing = replay(embodiment, sample, path.parent / trial["actions"])
            if success is None or float(success) != scores["success_at_end"]:
                raise SystemExit(f"{path} {sample['scene_id']}: replay does not reproduce")
            scores["lifted"] = float(lifted)
            scores["standing"] = float(standing)
        for name in ("lifted", "standing"):
            scored = [scores[name] for scores in sample["epochs"] if scores]
            if scored:
                sample["reduced"][name] = sum(scored) / len(scored)
    for name in ("lifted", "standing"):
        reduced = [s["reduced"][name] for s in log["samples"] if name in s["reduced"]]
        log["results"]["metrics"][name] = sum(reduced) / len(reduced)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", type=Path, nargs="+", help="Inspect Robots eval logs")
    args = parser.parse_args()

    embodiment = SimEmbodiment(until_done=True)
    try:
        logs = [(path, rescore(embodiment, path)) for path in args.logs]
    finally:
        embodiment.close()
    for path, log in logs:
        original = path.with_suffix(".json.orig")
        if not original.exists():
            shutil.copy2(path, original)
        path.write_text(json.dumps(log, indent=2))
        metrics = log["results"]["metrics"]
        print(f"{path}: lifted {metrics['lifted']:.2f}, standing {metrics['standing']:.2f}")


if __name__ == "__main__":
    main()
