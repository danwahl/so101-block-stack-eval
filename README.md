# so101-block-stack-eval

The SO-101 block stack in ManiSkill, as an [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment and task.

## Leaderboard

<!-- leaderboard:start -->

| # | Name | Provider | Stacked | Lifted | Cost | Calls | Tokens in | Tokens out | Minutes |
|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 1 | deepseek-v4.1-flash | deepseek | **70% (58%–82%)** | **90%** | 0.065 | 49 | 544,810 | 67,682 | 10.2 |
| 2 | glm-5.3-flash | z-ai | 40% (26%–54%) | 64% | 0.030 | 49 | 468,392 | 6,825 | 2.8 |
| 3 | gpt-6-luna | openai | 22% (12%–34%) | 34% | **0.011** | **21** | **118,672** | **3,125** | **1.1** |

50 layouts, one trial each; 95% bootstrap interval over layouts. Cost in US dollars, calls, tokens and minutes waiting on the model are means per trial. Prompt version `8dd3b142780d`.

<!-- leaderboard:end -->

Each model on layout 42, "stack the red block on the green block", from the top camera at twice real time:

| gpt-6-luna | deepseek-v4.1-flash | glm-5.3-flash |
|:-:|:-:|:-:|
| ![gpt-6-luna on layout 42](images/gpt-6-luna-layout-042.webp) | ![deepseek-v4.1-flash on layout 42](images/deepseek-v4.1-flash-layout-042.webp) | ![glm-5.3-flash on layout 42](images/glm-5.3-flash-layout-042.webp) |
| 68 steps | 80 steps | 282 steps |

The trials behind the table, with each run's actions, transcripts and model calls, are in the [Hugging Face dataset](https://huggingface.co/datasets/drwahl/so101-block-stack-eval).

## Task

`so101-block-stack` has one scene per layout in `src/so101_block_stack_eval/layouts.jsonl`. Each layout gives:
- the three blocks' positions and yaws;
- which block goes on which, and the instruction naming them;
- the arm's opening joint positions.

A trial succeeds on the first step the named block sits on its target, out of the jaws, with neither of the other blocks pushed out of place. A second scorer, `lifted`, records whether the jaws ever held the named block off the table.

## Embodiment

`so101_sim` is one ManiSkill environment with physics on the CPU. Each action is an absolute pose for the grip point: x, y and z in the arm's base frame, the wrist's yaw, the tool's pitch below level, and the jaw opening. The grip point is where a block's centre sits once the jaws close on it. A closed-form IK turns each pose into joint targets. A target the arm cannot reach, or one that would turn a joint too far in one step, is refused and the arm holds its last target. Observations are the wrist and overhead camera images, the commanded pose read back from the arm, and the joint positions.

## Running an agent

```
uv sync
export OPENROUTER_API_KEY=...
uv run python scripts/run.py z-ai/glm-5.3-flash
```

`scripts/run.py` passes the agent two things the `inspect-robots run` CLI cannot: `SimEmbodiment.pre_check`, which gives the reason a motion would be refused, and `LOCATE_POINTS`, a custom tool that maps top-camera pixels to base-frame points on a horizontal plane. Every model goes through OpenRouter at low reasoning effort.

Only the cameras render on the GPU. To run without one, hide it and render with Mesa's software Vulkan:

```
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/run.py z-ai/glm-5.3-flash
```

The agent plugin's custom tools are not released yet, so `inspect-robots-agent` and `inspect-robots` come from the `agent-custom-tools` branch of [a fork](https://github.com/danwahl/inspect-robots); `uv.lock` records the commit.

## Tests

```
uv run pytest
```

## License

Apache 2.0. The arm's description in `src/so101_block_stack_eval/description/` is adapted from the SO-101 in [TheRobotStudio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), under the same license.

## Citation

```bibtex
@misc{wahl2026so101blockstackeval,
  title = {{so101-block-stack-eval: LLM agents stacking blocks with a simulated SO-101 arm}},
  author = {Wahl, Dan},
  year = {2026},
  url = {https://github.com/danwahl/so101-block-stack-eval},
  note = {Version 0.1.0}
}
```
