"""Animate a trial as a WebP by replaying its logged actions through the sim.

    python scripts/render.py outputs/openai/gpt-6-luna/*-until-done_*.json layout-000

The sim is deterministic, so a log's actions reproduce the trial without storing frames.
An agent's trial is shown up to its last action, without the arm cleared away. A VLA never
ends a trial itself, so its trial is shown up to the first step the block sits stacked out
of the jaws, or to the last if it never does. Each frame is the top camera, at half size,
one per step.
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


def frames(embodiment, sample, actions_path, until_success):
    """The top camera at the start and after each logged action, stopping at the first
    success if ``until_success``."""
    scene = Scene(
        id=sample["scene_id"], instruction=sample["instruction"], metadata=sample["scene_metadata"]
    )
    yield embodiment.reset(scene).images["top"]
    with actions_path.open() as file:
        records = (json.loads(line) for line in file)
        actions = [record["action"] for record in records if "action" in record]
    for action in actions:
        result = embodiment.step(Action(data=np.asarray(action, np.float64)))
        yield result.observation.images["top"]
        if until_success and result.terminated:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="an Inspect Robots eval log")
    parser.add_argument("scene", help="scene id, such as layout-000")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--out", type=Path, help="defaults to images/<model>-<scene>.webp")
    args = parser.parse_args()

    log = json.loads(args.log.read_text())
    sample = next(s for s in log["samples"] if s["scene_id"] == args.scene)
    actions = args.log.parent / sample["trial_metadata"][args.epoch]["actions"]
    model = name(log["eval"]).split("/")[-1]
    out = args.out or Path("images") / f"{model}-{args.scene}.webp"

    # A VLA commands joint targets; an agent, the grip point.
    vla = log["eval"]["policy"] == "lerobot"
    control = "joints" if vla else "grip"
    embodiment = SimEmbodiment(control=control)
    try:
        images = [
            Image.fromarray(np.asarray(image)).resize((SIZE, SIZE), Image.LANCZOS)
            for image in frames(embodiment, sample, actions, until_success=vla)
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
    print(f"Wrote {out} ({len(images) - 1} steps, {out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
