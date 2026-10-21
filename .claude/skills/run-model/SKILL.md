---
name: run-model
description: Run the block stack on a model, retry any errored trials, regenerate the leaderboard, and suggest a commit
disable-model-invocation: true
argument-hint: [openrouter-or-anthropic-model-id]
---

# Run Model Evaluation

Run the until-done block stack for a model and update the leaderboard.

The argument is an OpenRouter model ID (e.g. `z-ai/glm-5.3-flash`), or an Anthropic model ID with an `anthropic/` prefix (e.g. `anthropic/claude-haiku-5-5`), which runs on Anthropic's own API. Logs go to `outputs/<provider>/<model>/`. Every command that builds the sim takes the prefix `CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`, which keeps it on the CPU.

## Steps

### 1. Check the OpenRouter balance

`scripts/run.py` reads `OPENROUTER_API_KEY` from the environment, and `.env` (gitignored) holds it. Load it in each shell that runs a step, and never print the key itself.

```bash
set -a; . ./.env; set +a
curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

The remaining balance is `data.total_credits - data.total_usage`. A trial has cost between $0.01 and $0.08, so 50 layouts need up to about $4, and runs in parallel draw on the same balance. Every call reserves the model's full output limit, so a low balance refuses calls (HTTP 402) before it is spent, and each refused call errors a trial. If the balance looks short, tell the user before starting.

For an `anthropic/` model, `scripts/run.py` reads `ANTHROPIC_API_KEY` instead, also from `.env`. Anthropic has no balance endpoint for an API key, so tell the user the estimate rather than checking it: a trial on Claude Haiku 5.5 has cost about $0.01, and the larger models cost more in proportion to their prices in `PRICES` in `scripts/analysis.py`. A model missing from `PRICES` stops `scripts/analysis.py`, so add its prices from Anthropic's pricing page before running it.

### 2. Run the evaluation

```bash
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/run.py $ARGUMENTS --until-done
```

Run it in the background, with `.env` loaded as in step 1. It takes one and a half to eight hours, depending on the model. After a few minutes, check that `outputs/<provider>/<model>/wire/<run>/layout-000-e0/calls.jsonl` exists and its lines have `"status": 200`; a trial's actions are written only when it ends. Then wait for the run to exit. Its last line is the `EvalResults`, with `errored_trials`, and the log is the newest `so101-block-stack-until-done_*.json` in the model's directory.

### 3. Retry errored trials

If any trial errored, list the errored samples in the log (`status` other than `"success"`) with their `error`:

- From the service, to rerun: `LLM request rejected — HTTP 402` (OpenRouter credits), `LLM request rejected — HTTP 400` naming a low credit balance (Anthropic credits), and `LLM request failed after 3 attempts` (rate limits, server errors, timeouts).
- From the model, to keep: `LLM tool calls kept failing`, `LLM produced no tool call in 3 consecutive turns`, and `LLM refused the request`.

Show any other error to the user rather than guessing. Rerun the service errors and keep the model's, so no model gets a second attempt:

```bash
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/retry.py <log> --keep layout-NNN
```

Pass `--keep` once per scene to keep. The retry writes its own log of the rerun scenes, then the combined log, and prints the combined log's path as its last line. If the combined log still has service errors, retry it in turn with the same `--keep` flags, until only kept scenes are errored.

### 4. Regenerate the leaderboard

```bash
uv run python scripts/analysis.py
```

This counts each model's latest complete until-done run and rewrites `results.json` and the leaderboard between the `<!-- leaderboard:start -->` and `<!-- leaderboard:end -->` markers in README.md. It stops if the runs saw different prompts, or if two providers share a model name; if so, tell the user rather than rerunning the other models. Check the new model's row.

### 5. Render layout 42

```bash
CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    uv run python scripts/render.py <combined log> layout-042
```

This writes `images/<model>-layout-042.webp`. Add the model's column to the "Each model on layout 42" table in README.md, in leaderboard order, with its step count: the frame count `render.py` prints, less one.

### 6. Suggest a commit

Show the user a suggested commit in the style of existing commits, with a body giving the new row:

```
feat: add <Model Display Name> to the leaderboard

<model>  <Stacked>% (<lo>-<hi>%)   lifted <Lifted>%
```

Use the model's common display name rather than the raw ID. Stage `README.md`, `results.json` and the webp. Don't commit automatically; suggest the command and let the user decide. The push redeploys the site.

### 7. Suggest a dataset update

Once the user has pushed the commit, suggest updating the [Hugging Face dataset](https://huggingface.co/datasets/drwahl/so101-block-stack-eval):

```bash
uv run python scripts/export.py && uv run hf upload drwahl/so101-block-stack-eval dataset . --repo-type dataset --delete "logs/*" --commit-message "Add <Model Display Name> (danwahl/so101-block-stack-eval@<short commit>)"
```

The export rewrites `dataset/` from the counted runs. The upload skips unchanged files and deletes logs of runs no longer counted. Don't upload automatically; suggest the command and let the user decide.
