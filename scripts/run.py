"""Score an LLM agent on the block stack in sim.

    python scripts/run.py z-ai/glm-5.3-flash --until-done

The CLI runs the same thing (`inspect-robots run --task so101-block-stack --policy agent
--embodiment so101_sim -P model=...`) but cannot pass the agent callables: the reach check,
without which a move out of reach stalls the arm instead of returning a reason, and the
locate_points tool.

Models are named as OpenRouter names them, and runs go through OpenRouter with
OPENROUTER_API_KEY, so each model is sent the same request format and each call reports
its cost. Anthropic's models, named as Anthropic names them (anthropic/claude-haiku-5-5),
go instead to Anthropic's Messages API with ANTHROPIC_API_KEY, and analysis.py prices
their calls from the tokens.

Physics runs on the CPU; only the cameras render on the GPU. To keep a run off the GPU,
hide it and render with Mesa's software Vulkan, at about 0.1 s a step:

    CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
        python scripts/run.py z-ai/glm-5.3-flash --until-done

Logs go to outputs/<provider>/<model>/. The leaderboard counts --until-done runs over the
first 50 layouts, the default.
"""

import argparse

from inspect_robots import eval as ir_eval
from inspect_robots_agent import LLMAgentPolicy

from so101_block_stack_eval.embodiment import LOCATE_POINTS, SimEmbodiment
from so101_block_stack_eval.task import stack

OPENROUTER = "https://openrouter.ai/api/v1"
ANTHROPIC = "https://api.anthropic.com/v1"


def agent(model, embodiment, *, max_llm_calls=100, effort="low", locate=True):
    """The LLM agent, its moves checked against ``embodiment``'s reach."""
    if model.startswith("anthropic/"):
        route = {
            "model": model.removeprefix("anthropic/"),
            "base_url": ANTHROPIC,
            "api_key_env": "ANTHROPIC_API_KEY",
            "wire": "messages",
        }
    else:
        route = {"model": model, "base_url": OPENROUTER}
    return LLMAgentPolicy(
        **route,
        max_llm_calls=max_llm_calls,
        pre_check=embodiment.pre_check,
        tools=(LOCATE_POINTS,) if locate else (),
        effort=effort,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="provider/model, as the agent plugin names it")
    parser.add_argument("--episodes", type=int, default=50, help="first N held-out layouts")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--effort", default="low", help="reasoning effort")
    parser.add_argument("--log-dir", help="defaults to outputs/<provider>/<model>")
    parser.add_argument(
        "--no-locate", action="store_true", help="withhold the locate_points tool from the agent"
    )
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="run each trial on past a stack until the agent stops it or the steps run out, "
        "then clear the arm away and score the stack left standing",
    )
    args = parser.parse_args()

    embodiment = SimEmbodiment(until_done=args.until_done)
    try:
        (log,) = ir_eval(
            stack(episodes=args.episodes, max_steps=args.max_steps, until_done=args.until_done),
            agent(
                args.model,
                embodiment,
                max_llm_calls=args.max_llm_calls,
                effort=args.effort,
                locate=not args.no_locate,
            ),
            embodiment,
            log_dir=args.log_dir or f"outputs/{args.model}",
        )
    finally:
        embodiment.close()
    print(log.results)


if __name__ == "__main__":
    main()
