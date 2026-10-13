"""Score an LLM agent on the block stack in sim.

    python scripts/run.py z-ai/glm-5.3-flash --episodes 30

The CLI runs the same thing (`inspect-robots run --task so101-block-stack --policy agent
--embodiment so101_sim -P model=...`) but cannot pass the agent callables: the reach check,
without which a move out of reach stalls the arm instead of returning a reason, and the
locate_points tool.

The model's API key comes from the environment. With OPENROUTER_API_KEY set, a model is
named as OpenRouter names it; a provider's own key (ANTHROPIC_API_KEY, ...) sends that
provider's models to it directly.

Physics runs on the CPU; only the cameras render on the GPU. To keep a run off the GPU,
hide it and render with Mesa's software Vulkan, at about 0.1 s a step:

    CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
        python scripts/run.py z-ai/glm-5.3-flash --episodes 30
"""

import argparse

from inspect_robots import eval as ir_eval
from inspect_robots_agent import LLMAgentPolicy

from so101_block_stack.embodiment import LOCATE_POINTS, SimEmbodiment
from so101_block_stack.task import stack


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="provider/model, as the agent plugin names it")
    parser.add_argument("--episodes", type=int, default=30, help="first N held-out layouts")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--effort", help="reasoning effort; the provider's default if unset")
    parser.add_argument("--log-dir", default="outputs")
    parser.add_argument(
        "--no-locate", action="store_true", help="withhold the locate_points tool from the agent"
    )
    args = parser.parse_args()

    embodiment = SimEmbodiment()
    try:
        (log,) = ir_eval(
            stack(episodes=args.episodes, max_steps=args.max_steps),
            LLMAgentPolicy(
                model=args.model,
                max_llm_calls=args.max_llm_calls,
                pre_check=embodiment.pre_check,
                tools=() if args.no_locate else (LOCATE_POINTS,),
                **({"effort": args.effort} if args.effort else {}),
            ),
            embodiment,
            log_dir=args.log_dir,
            store_frames=True,
        )
    finally:
        embodiment.close()
    print(log.results)


if __name__ == "__main__":
    main()
