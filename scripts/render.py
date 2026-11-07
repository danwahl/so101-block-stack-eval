"""Animate a trial as a WebP by replaying its logged actions through the sim.

    python scripts/render.py outputs/openai/gpt-6-luna/*-until-done_*.json layout-000

The sim is deterministic, so a log's actions reproduce the trial without storing frames.
An until-done trial is shown up to the agent's last action, without the arm cleared away;
``--steps`` cuts it shorter, for a VLA, which never ends a trial itself and so always runs
to the task's last step. Each frame is the top camera, at half size, one per step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis import name
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from PIL import Image

from so101_block_stack_eval.embodiment import SimEmbodiment

SIZE = 240
FRAME_MS = 100
HOLD_MS = 1500


def frames(embodiment, sample, actions_path, steps=None):
    """The top camera at the start and after each logged action, up to ``steps`` of them."""
    scene = Scene(
        id=sample["scene_id"], instruction=sample["instruction"], metadata=sample["scene_metadata"]
    )
    yield embodiment.reset(scene).images["top"]
    with actions_path.open() as file:
        records = (json.loads(line) for line in file)
        actions = [record["action"] for record in records if "action" in record]
    for action in actions[:steps]:
        yield embodiment.step(Action(data=np.asarray(action, np.float64))).observation.images["top"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="an Inspect Robots eval log")
    parser.add_argument("scene", help="scene id, such as layout-000")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--steps", type=int, help="show only the first N steps")
    parser.add_argument("--out", type=Path, help="defaults to images/<model>-<scene>.webp")
    args = parser.parse_args()

    log = json.loads(args.log.read_text())
    sample = next(s for s in log["samples"] if s["scene_id"] == args.scene)
    actions = args.log.parent / sample["trial_metadata"][args.epoch]["actions"]
    model = name(log["eval"]).split("/")[-1]
    out = args.out or Path("images") / f"{model}-{args.scene}.webp"

    # A VLA commands joint targets; an agent, the grip point.
    control = "joints" if log["eval"]["policy"] == "lerobot" else "grip"
    embodiment = SimEmbodiment(control=control)
    try:
        images = [
            Image.fromarray(np.asarray(image)).resize((SIZE, SIZE), Image.LANCZOS)
            for image in frames(embodiment, sample, actions, args.steps)
        ]
    finally:
        embodiment.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=[FRAME_MS] * (len(images) - 1) + [HOLD_MS],
        loop=0,
        quality=70,
    )
    print(f"Wrote {out} ({len(images)} frames, {out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
