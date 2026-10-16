"""Animate a trial as a WebP by replaying its logged actions through the sim.

    python scripts/render.py outputs/openai/gpt-6-luna/*-until-done_*.json layout-000

The sim is deterministic, so a log's actions reproduce the trial without storing frames.
An until-done trial's last frame is the arm cleared away.
Each frame is the top camera, at half size, every second step and the last: twice real time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from PIL import Image

from so101_block_stack_eval.embodiment import SimEmbodiment

STRIDE = 2
SIZE = 240
FRAME_MS = 100
HOLD_MS = 1500


def frames(embodiment, sample, actions_path):
    """The top camera at the start and after each logged action."""
    scene = Scene(
        id=sample["scene_id"], instruction=sample["instruction"], metadata=sample["scene_metadata"]
    )
    yield embodiment.reset(scene).images["top"]
    with actions_path.open() as file:
        actions = [json.loads(line)["action"] for line in file if '"action"' in line]
    for t, action in enumerate(actions):
        meta = {"request_stop": True} if t == len(actions) - 1 else {}
        action = Action(data=np.asarray(action, np.float64), meta=meta)
        yield embodiment.step(action).observation.images["top"]


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
    model = log["eval"]["policy_config"]["model"].split("/")[-1]
    out = args.out or Path("images") / f"{model}-{args.scene}.webp"

    embodiment = SimEmbodiment(until_done=log["eval"]["task"].endswith("-until-done"))
    try:
        images, last = [], None
        for i, image in enumerate(frames(embodiment, sample, actions)):
            last = Image.fromarray(np.asarray(image)).resize((SIZE, SIZE), Image.LANCZOS)
            if i % STRIDE == 0:
                images.append(last)
        if images[-1] is not last:
            images.append(last)
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
