"""Build the leaderboard from the eval logs under outputs/.

    python scripts/analysis.py

Each model's latest complete run of the until-done task over the first EPISODES layouts
counts, one trial per layout, at low reasoning effort through OpenRouter's chat API, or
Anthropic's Messages API for Anthropic's models. OpenRouter reports each call's cost;
Anthropic's calls are priced from their tokens at PRICES. A trial counts as stacked if its
``standing`` score holds. A trial that ended in an error, such as three failed turns in a
row, counts as neither stacked nor lifted. The 95% bootstrap intervals resample layouts.
Every counted trial must have seen the same system prompt and tool schemas; their hash is
the prompt version, written to results.json and the README with the leaderboard.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from evalib import Column, Leaderboard, bootstrap, provider, update_readme

TASK = "so101-block-stack-until-done"
EPISODES = 50
LAYOUTS = [f"layout-{i:03d}" for i in range(EPISODES)]
OPENROUTER = "https://openrouter.ai/api/v1"
ANTHROPIC = "https://api.anthropic.com/v1"
ROUTES = {(OPENROUTER, "chat"), (ANTHROPIC, "messages")}
# US dollars per million tokens of input, 5-minute cache writes, cache hits and output, from
# https://platform.claude.com/docs/en/about-claude/pricing. Claude Haiku 5.5 costs more for
# a call whose prompt is over 100,000 tokens.
PRICES = {
    "claude-fable-5-1": (10.0, 12.5, 0.25, 50.0),
    "claude-opus-5-5": (4.0, 5.0, 0.2, 20.0),
    "claude-sonnet-5-5": (2.0, 2.5, 0.1, 10.0),
    "claude-haiku-5-5": (0.1, 0.125, 0.01, 0.5),
}
LONG_PRICES = {"claude-haiku-5-5": (100_000, (0.5, 0.625, 0.05, 2.5))}

COLUMNS = [
    Column(
        "standing",
        "Stacked",
        "higher",
        "pct",
        "Layouts where the stack stands once the arm is cleared away",
    ),
    Column(
        "lifted",
        "Lifted",
        "higher",
        "pct",
        "Layouts where the jaws held the named block off the table",
    ),
    Column("cost", "Cost", "lower", ".3f", "Mean model cost per trial, in US dollars"),
    Column("calls", "Calls", "lower", ".0f", "Mean model calls per trial"),
    Column("tokens_in", "Tokens in", "lower", ",.0f", "Mean prompt tokens per trial"),
    Column(
        "tokens_out",
        "Tokens out",
        "lower",
        ",.0f",
        "Mean completion tokens per trial, reasoning included",
    ),
    Column("minutes", "Minutes", "lower", ".1f", "Mean time per trial spent waiting on the model"),
]


def model_id(config: dict) -> str:
    """The provider/model name a run was given, from its logged policy config: the agent
    plugin logs an Anthropic model without its provider."""
    if config["base_url"] == ANTHROPIC:
        return f"anthropic/{config['model']}"
    return config["model"]


def latest_runs(root: Path):
    """Each model's latest complete run over the first EPISODES layouts, with its path."""
    runs = {}
    for path in sorted(root.rglob("*.json")):
        log = json.loads(path.read_text())
        if not isinstance(log, dict) or "eval" not in log:
            continue
        spec, config = log["eval"], log["eval"]["policy_config"]
        if spec["task"] != TASK or log["status"] != "success":
            continue
        if (config["base_url"], config["wire"]) not in ROUTES or config["effort"] != "low":
            continue
        if sorted(s["scene_id"] for s in log["samples"]) != LAYOUTS:
            continue
        if any(len(s["trial_metadata"]) != 1 for s in log["samples"]):
            continue
        model = model_id(config)
        if model not in runs or spec["created"] > runs[model][1]["eval"]["created"]:
            runs[model] = (path, log)
    return runs.values()


def fingerprint(directory: Path, trial: dict) -> str | None:
    """A hash of the system prompt and tool schemas the trial's first call sent."""
    if "wire_capture" not in trial:
        return None
    with (directory / trial["wire_capture"]).open() as file:
        request = json.loads(file.readline())["request"]
    if "system" in request:
        # The Messages API's shape, put back in the chat API's so both hash alike.
        system = request["system"][0]["text"]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in request["tools"]
        ]
    else:
        system = next(m["content"] for m in request["messages"] if m["role"] == "system")
        tools = request["tools"]
    text = json.dumps({"system": system, "tools": tools}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def call_usage(model: str, call: dict) -> tuple[float, int, int]:
    """One call's cost in US dollars and its prompt and completion tokens."""
    reported = (call.get("response") or {}).get("usage") or {}
    if call.get("endpoint") != "/messages":
        return (
            reported.get("cost", 0.0),
            reported.get("prompt_tokens", 0),
            reported.get("completion_tokens", 0),
        )
    tokens = [
        reported.get(key, 0)
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        )
    ]
    if model not in PRICES:
        raise SystemExit(f"no price for {model}: add it to PRICES")
    prices = PRICES[model]
    threshold, long_prices = LONG_PRICES.get(model, (None, None))
    if threshold is not None and sum(tokens[:3]) > threshold:
        prices = long_prices
    cost = sum(n * price for n, price in zip(tokens, prices, strict=True)) / 1e6
    return cost, sum(tokens[:3]), tokens[3]


def trial_usage(directory: Path, trial: dict, model: str) -> dict[str, float]:
    """The trial's model calls, tokens, cost and model time, from its wire log."""
    usage = {"cost": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "minutes": 0.0}
    if "wire_capture" not in trial:
        return usage
    with (directory / trial["wire_capture"]).open() as file:
        for line in file:
            call = json.loads(line)
            cost, tokens_in, tokens_out = call_usage(model, call)
            usage["cost"] += cost
            usage["calls"] += 1
            usage["tokens_in"] += tokens_in
            usage["tokens_out"] += tokens_out
            usage["minutes"] += call["duration_s"] / 60
    return usage


def load_trials(root: Path) -> pd.DataFrame:
    """One row per trial of every counted run."""
    rows = []
    for path, log in latest_runs(root):
        model = model_id(log["eval"]["policy_config"])
        for sample in log["samples"]:
            for epoch, (scores, trial) in enumerate(
                zip(sample["epochs"], sample["trial_metadata"], strict=True)
            ):
                rows.append(
                    {
                        "model": model.split("/")[-1],
                        "provider": provider(model),
                        "run": path.stem,
                        "layout": sample["scene_id"],
                        "epoch": epoch,
                        "standing": scores.get("standing", 0.0),
                        "lifted": scores.get("lifted", 0.0),
                        "errored": float(not scores),
                        **trial_usage(path.parent, trial, model.split("/")[-1]),
                        "prompt": fingerprint(path.parent, trial),
                    }
                )
    return pd.DataFrame(rows)


def scores(trials: pd.DataFrame) -> dict[str, float]:
    return {column.key: trials[column.key].mean() for column in COLUMNS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates")
    args = parser.parse_args()

    trials = load_trials(args.logs)
    if trials.empty:
        raise SystemExit(f"no complete {EPISODES}-layout runs under {args.logs}")
    names = trials.groupby("model")["provider"].nunique()
    if (names > 1).any():
        raise SystemExit(f"model names shared across providers: {list(names[names > 1].index)}")
    prompts = trials["prompt"].dropna().unique()
    if len(prompts) != 1:
        by_model = trials.groupby("model")["prompt"].unique().to_dict()
        raise SystemExit(f"runs saw different prompts: {by_model}")

    board = Leaderboard(
        name=TASK,
        columns=COLUMNS,
        info={"provider": "Provider"},
        notes=[
            f"{EPISODES} layouts, one trial each; 95% bootstrap interval over layouts. "
            "Cost in US dollars, calls, tokens and minutes waiting on the model are means "
            "per trial. "
            f"Prompt version `{prompts[0]}`."
        ],
    )
    board.add(
        bootstrap(trials, scores, by="model", cluster="layout", n=args.bootstrap),
        info=trials.groupby("model")[["provider"]].first(),
    )
    board.save(args.results)
    update_readme(board.markdown(), args.readme)
    print(board.markdown())


if __name__ == "__main__":
    main()
