"""Build the leaderboard from the eval logs under outputs/.

    python scripts/analysis.py

Each model's latest complete run of the until-done task over the first EPISODES layouts
counts, one trial per layout, through OpenRouter's chat API at low reasoning effort. A
trial counts as stacked if its ``standing`` score holds. A trial that ended in an error,
such as three failed turns in a row, counts as neither stacked nor lifted. The 95%
bootstrap intervals resample layouts. Every counted trial must have seen the same system
prompt and tool schemas; their hash is the prompt version, written to results.json and
the README with the leaderboard.
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
        if (config["base_url"], config["wire"], config["effort"]) != (OPENROUTER, "chat", "low"):
            continue
        if sorted(s["scene_id"] for s in log["samples"]) != LAYOUTS:
            continue
        if any(len(s["trial_metadata"]) != 1 for s in log["samples"]):
            continue
        model = config["model"]
        if model not in runs or spec["created"] > runs[model][1]["eval"]["created"]:
            runs[model] = (path, log)
    return runs.values()


def fingerprint(directory: Path, trial: dict) -> str | None:
    """A hash of the system prompt and tool schemas the trial's first call sent."""
    if "wire_capture" not in trial:
        return None
    with (directory / trial["wire_capture"]).open() as file:
        request = json.loads(file.readline())["request"]
    system = next(m["content"] for m in request["messages"] if m["role"] == "system")
    text = json.dumps({"system": system, "tools": request["tools"]}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def trial_usage(directory: Path, trial: dict) -> dict[str, float]:
    """The trial's model calls, tokens, cost and model time, from its wire log."""
    usage = {"cost": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "minutes": 0.0}
    if "wire_capture" not in trial:
        return usage
    with (directory / trial["wire_capture"]).open() as file:
        for line in file:
            call = json.loads(line)
            reported = (call.get("response") or {}).get("usage") or {}
            usage["cost"] += reported.get("cost", 0.0)
            usage["calls"] += 1
            usage["tokens_in"] += reported.get("prompt_tokens", 0)
            usage["tokens_out"] += reported.get("completion_tokens", 0)
            usage["minutes"] += call["duration_s"] / 60
    return usage


def load_trials(root: Path) -> pd.DataFrame:
    """One row per trial of every counted run."""
    rows = []
    for path, log in latest_runs(root):
        model = log["eval"]["policy_config"]["model"]
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
                        **trial_usage(path.parent, trial),
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
