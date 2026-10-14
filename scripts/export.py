"""Gather the counted runs into dataset/ for publishing on Hugging Face.

    python scripts/export.py
    hf upload drwahl/so101-block-stack-eval dataset . --repo-type dataset --delete "logs/*"

dataset/data/trials.parquet holds one row per trial, as scripts/analysis.py scores it,
and dataset/logs/<provider>/<model>/ each counted run's eval log with the actions,
transcripts and wire calls beside it. The wire calls, with the images in their blobs/
directory, are the record of exactly what the model was sent.
"""

import argparse
import shutil
from pathlib import Path

from analysis import latest_runs, load_trials

CARD = """\
---
license: apache-2.0
configs:
  - config_name: default
    data_files: data/trials.parquet
---

# so101-block-stack-eval

Agent trials from [so101-block-stack-eval](https://github.com/danwahl/so101-block-stack-eval):
an LLM driving a simulated SO-101 arm to stack one named block on another.

`data/trials.parquet` has one row per trial: the model, the layout, whether the block
ended stacked or was ever lifted, whether the trial ended in an error, its model cost in
US dollars, and the hash of the prompt it was sent. `logs/<provider>/<model>/` holds each
run's Inspect Robots eval log, with each trial's actions, transcript, and wire calls (the
requests and responses, with images under `wire/<run>/blobs/`).
"""


def trials(log):
    return [trial for sample in log["samples"] for trial in sample["trial_metadata"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    shutil.rmtree(args.out / "logs", ignore_errors=True)
    for path, log in latest_runs(args.logs):
        run = path.parent
        target = args.out / "logs" / log["eval"]["policy_config"]["model"]
        target.mkdir(parents=True)
        shutil.copy2(path, target / path.name)
        for trial in trials(log):
            for key in ("actions", "transcript", "wire_capture"):
                if key in trial:
                    (target / trial[key]).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(run / trial[key], target / trial[key])
        runs = {Path(t["wire_capture"]).parts[1] for t in trials(log) if "wire_capture" in t}
        for run_id in runs:
            blobs = Path("wire") / run_id / "blobs"
            if (run / blobs).is_dir():
                shutil.copytree(run / blobs, target / blobs)

    (args.out / "data").mkdir(parents=True, exist_ok=True)
    rows = load_trials(args.logs)
    rows.to_parquet(args.out / "data" / "trials.parquet", index=False)
    (args.out / "README.md").write_text(CARD)
    print(f"Wrote {len(rows)} trials to {args.out}")


if __name__ == "__main__":
    main()
