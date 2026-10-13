# inspect-robots-so101-block-stack

The SO-101 block stack in ManiSkill, as an [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment and task. An SO-101 arm stands at a table with three 30 mm cubes, red, green and blue, and is asked to stack one named block on another. A wrist camera and an overhead camera see the table.

## Task

`so101-block-stack` holds 150 layouts, one scene each, in `src/so101_block_stack/layouts.jsonl`. Each layout gives:
- the three blocks' positions and yaws;
- which block goes on which, and the instruction naming them;
- the arm's opening joint positions.

A trial succeeds on the first step the named block sits on its target, out of the jaws, with neither of the other blocks moved more than 15 mm. A second scorer, `lifted`, records whether the named block was ever lifted 5 cm off the table. Trials run to 600 steps at 10 Hz unless `max_steps` says otherwise.

## Embodiment

`so101_sim` is one ManiSkill environment with physics on the CPU. Each action is an absolute pose for the grip point: x, y and z in the arm's base frame, the wrist's yaw, the tool's pitch below level, and the jaw opening. The grip point is where a block's centre sits once the jaws close on it. A closed-form IK turns each pose into joint targets. A target the arm cannot reach, or one that would turn a joint more than 10 degrees in one step, is refused and the arm holds its last target. Observations are both 480 x 480 camera images, the commanded pose read back from the arm, and the joint positions.

`SimEmbodiment.pre_check` gives the reason a motion would be refused, and `LOCATE_POINTS` is a custom tool for the agent policy. It maps top-camera pixels to base-frame points on a horizontal plane.

## Running an agent

```
uv sync
export OPENROUTER_API_KEY=...
uv run python scripts/run.py z-ai/glm-5.3-flash --episodes 30
```

`scripts/run.py` passes the agent the embodiment's `pre_check` and `locate_points`; the `inspect-robots run` CLI cannot pass either. Without `pre_check`, a move out of reach stalls the arm instead of returning a reason. Logs go under `outputs/`, and `scripts/cost.py outputs` totals what the model calls cost.

Only the cameras render on the GPU. To run without one, hide it and render with Mesa's software Vulkan, at about 0.1 s a step:

```
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/run.py z-ai/glm-5.3-flash --episodes 30
```

The agent plugin's custom tools are not released yet, so `inspect-robots-agent` and `inspect-robots` come from a commit on [a fork](https://github.com/danwahl/inspect-robots).

The arm's description in `src/so101_block_stack/description/` is adapted from the SO-101 in [TheRobotStudio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), under the Apache 2.0 license.

## Tests

```
uv run pytest
```
