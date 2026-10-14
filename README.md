# so101-block-stack-eval

The SO-101 block stack in ManiSkill, as an [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment and task. An SO-101 arm stands at a table with three 30 mm cubes, red, green and blue, and is asked to stack one named block on another. A wrist camera and an overhead camera see the table.

## Leaderboard

<!-- leaderboard:start -->

| # | Name | Provider | Stacked | Lifted | Errored | Cost | Calls | Tokens in | Tokens out | Minutes |
|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | deepseek-v4.1-flash | deepseek | **70% (58%–82%)** | **90%** | 0% | 3.27 | 49 | 544,810 | 67,682 | 10.2 |
| 2 | glm-5.3-flash | z-ai | 40% (26%–54%) | 64% | 0% | 1.52 | 49 | 468,392 | 6,825 | 2.8 |
| 3 | gpt-6-luna | openai | 22% (12%–34%) | 34% | 0% | **0.53** | **21** | **118,672** | **3,125** | **1.1** |

50 layouts, one trial each; 95% bootstrap interval over layouts. Prompt version `8dd3b142780d`.

<!-- leaderboard:end -->

## Task

`so101-block-stack` holds 150 layouts, one scene each, in `src/so101_block_stack_eval/layouts.jsonl`. Each layout gives:
- the three blocks' positions and yaws;
- which block goes on which, and the instruction naming them;
- the arm's opening joint positions.

A trial succeeds on the first step the named block sits on its target, out of the jaws, with neither of the other blocks moved more than 15 mm. A second scorer, `lifted`, records whether the jaws ever held the named block off the table. Trials run to 600 steps at 10 Hz unless `max_steps` says otherwise.

## Embodiment

`so101_sim` is one ManiSkill environment with physics on the CPU. Each action is an absolute pose for the grip point: x, y and z in the arm's base frame, the wrist's yaw, the tool's pitch below level, and the jaw opening. The grip point is where a block's centre sits once the jaws close on it. A closed-form IK turns each pose into joint targets. A target the arm cannot reach, or one that would turn a joint more than 10 degrees in one step, is refused and the arm holds its last target. Observations are both 480 x 480 camera images, the commanded pose read back from the arm, and the joint positions.

`SimEmbodiment.pre_check` gives the reason a motion would be refused, and `LOCATE_POINTS` is a custom tool for the agent policy. It maps top-camera pixels to base-frame points on a horizontal plane.

## Running an agent

```
uv sync
export OPENROUTER_API_KEY=...
uv run python scripts/run.py z-ai/glm-5.3-flash
```

`scripts/run.py` passes the agent the embodiment's `pre_check` and `locate_points`; the `inspect-robots run` CLI cannot pass either. Without `pre_check`, a move out of reach stalls the arm instead of returning a reason. Every model goes through OpenRouter at low reasoning effort, so each is sent the same request format and each call reports its cost. A run covers the first 50 layouts and logs to `outputs/<provider>/<model>/`.

Only the cameras render on the GPU. To run without one, hide it and render with Mesa's software Vulkan, at about 0.1 s a step:

```
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/run.py z-ai/glm-5.3-flash
```

## Results

- `scripts/retry.py` reruns the scenes that errored in a run, as Inspect AI's `eval_retry` does, and writes a complete log that keeps the rest.
- `scripts/rescore.py` rewrites `lifted` in logs scored under an earlier rule, by replaying each trial's actions.
- `scripts/analysis.py` builds the leaderboard from each model's latest 50-layout run, with 95% bootstrap intervals over layouts. It writes `results.json` and the table above. It stops if the runs were sent different system prompts or tool schemas, and the table names the hash they share as the prompt version.
- `scripts/export.py` gathers the counted runs into `dataset/` for the [Hugging Face dataset](https://huggingface.co/datasets/drwahl/so101-block-stack-eval): a table of trials, and each run's log with its actions, transcripts and wire calls.
- `scripts/render.py` animates a trial as a WebP by replaying its logged actions through the sim.
- `scripts/cost.py` totals what the model calls under a log directory cost.

The agent plugin's custom tools are not released yet, so `inspect-robots-agent` and `inspect-robots` come from the `agent-custom-tools` branch of [a fork](https://github.com/danwahl/inspect-robots); `uv.lock` records the commit.

The arm's description in `src/so101_block_stack_eval/description/` is adapted from the SO-101 in [TheRobotStudio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), under the Apache 2.0 license.

## Tests

```
uv run pytest
```
