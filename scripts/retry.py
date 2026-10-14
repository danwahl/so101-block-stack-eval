"""Rerun the scenes that errored in an eval log and write a complete log with the rest kept.

    python scripts/retry.py outputs/z-ai/glm-5.3-flash/so101-block-stack_f265c43d.json

This follows Inspect AI's ``eval_retry``. The task and policy are rebuilt as run.py
builds them from what the log recorded, and the retry refuses to start if their scenes,
horizon, epochs or policy config differ from the log's, or to finish if the embodiment
does. Scenes logged as succeeded are kept as logged; any other scene runs again, every
epoch of it, under the seed it had. The retry's own log of those scenes is written as
eval() writes it, then the combined log beside it, and the original is left as it was.

The combined log's ``eval`` block is the retry's, saying when and from which commit it
ran. Its ``stats`` add up both runs, the discarded trials' steps included, with latency
weighted by steps; an empty epoch counts as errored, as eval() records one.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from statistics import mean

from inspect_robots import eval as ir_eval
from inspect_robots.log import EvalLog, EvalResults, EvalStats, read_eval_log
from inspect_robots.logging import JsonLogSink
from run import agent

from so101_block_stack_eval.embodiment import SimEmbodiment
from so101_block_stack_eval.task import stack


def as_json(value):
    return json.loads(json.dumps(value))


def merge(old: EvalLog, new: EvalLog) -> EvalLog:
    """``old`` with ``new``'s scenes in place of its own, and ``results`` recomputed the
    way ``eval()`` computes them."""
    retried = {sample.scene_id: sample for sample in new.samples}
    samples = tuple(retried.get(sample.scene_id, sample) for sample in old.samples)
    epochs = [scores for sample in samples for scores in sample.epochs]
    errored = sum(1 for scores in epochs if not scores)

    metrics = {}
    for name in dict.fromkeys(name for sample in samples for name in sample.reduced):
        voted = [s.reduced[name] for s in samples if s.reduced.get(name) is not None]
        metrics[name] = mean(voted) if voted else None
    abstentions = {}
    for scores in epochs:
        for name, value in scores.items():
            if value is None:
                abstentions[name] = abstentions.get(name, 0) + 1

    latencies = [
        (log.stats.mean_inference_latency_s, log.stats.total_steps)
        for log in (old, new)
        if log.stats.mean_inference_latency_s is not None
    ]
    stats = EvalStats(
        started_at=old.stats.started_at,
        completed_at=new.stats.completed_at,
        duration_s=old.stats.duration_s + new.stats.duration_s,
        total_steps=old.stats.total_steps + new.stats.total_steps,
        mean_inference_latency_s=(
            sum(latency * steps for latency, steps in latencies)
            / sum(steps for _, steps in latencies)
            if latencies
            else None
        ),
        frames_dir=new.stats.frames_dir or old.stats.frames_dir,
    )
    nothing_scored = errored == len(epochs)
    return EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status="error" if nothing_scored else "success",
        eval=new.eval,
        results=EvalResults(
            total_scenes=len(samples),
            total_trials=len(epochs),
            metrics=metrics,
            errored_trials=errored,
            abstentions=abstentions,
        ),
        stats=stats,
        samples=samples,
        error=f"all {len(epochs)} trial(s) errored; nothing was scored" if nothing_scored else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="an Inspect Robots eval log")
    args = parser.parse_args()

    old = read_eval_log(str(args.log))
    # Only a run that finished: a halted one is missing scenes, and the log does not
    # record how many it planned. The second form is the one eval() gives a finished run
    # in which every trial errored.
    finished = f"all {old.results.total_trials} trial(s) errored; nothing was scored"
    if old.status != "success" and old.error != finished:
        raise SystemExit(f"{args.log}: the run did not finish ({old.status}: {old.error})")
    errored = {sample.scene_id for sample in old.samples if sample.status != "success"}
    if not errored:
        print(f"{args.log}: every scene succeeded")
        return
    task = stack(episodes=len(old.samples), max_steps=old.eval.max_steps)
    logged = [(s.scene_id, s.instruction, s.scene_metadata) for s in old.samples]
    rebuilt = [(s.id, s.instruction, dict(s.metadata)) for s in task.scenes]
    if old.eval.task != task.name or as_json(logged) != as_json(rebuilt):
        raise SystemExit(f"{args.log}: the log's scenes are not this task's")
    if old.eval.max_seconds is not None or old.eval.grader is not None:
        raise SystemExit(f"{args.log}: the log's horizon or grader is not this task's")
    if any(len(sample.epochs) != task.epoch_spec.count for sample in old.samples):
        raise SystemExit(f"{args.log}: the log's epochs are not this task's")

    config = old.eval.policy_config
    embodiment = SimEmbodiment()
    try:
        policy = agent(
            config["model"],
            embodiment,
            max_llm_calls=config["max_llm_calls"],
            effort=config["effort"],
            locate=bool(config["tools"]),
        )
        if as_json(dataclasses.asdict(policy.config)) != as_json(config):
            raise SystemExit(f"{args.log}: the policy differs from the log's")
        (new,) = ir_eval(
            dataclasses.replace(task, scenes=[s for s in task.scenes if s.id in errored]),
            policy,
            embodiment,
            log_dir=str(args.log.parent),
            seed=old.eval.seed,
        )
    finally:
        embodiment.close()

    embodiment_fields = ("embodiment", "embodiment_info", "environment_id")
    if any(getattr(new.eval, f) != getattr(old.eval, f) for f in embodiment_fields):
        raise SystemExit(f"{args.log}: the retry's embodiment is not the log's")
    if new.status == "cancelled":
        raise SystemExit(f"{args.log}: the retry was cancelled; its own log holds what ran")
    sink = JsonLogSink(str(args.log.parent))
    log = merge(old, new)
    sink.on_eval_end(log)
    print(f"{sink.path}: {log.results}")


if __name__ == "__main__":
    main()
