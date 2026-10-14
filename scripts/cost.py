"""What the model calls under a log directory cost, from OpenRouter's own per-call figure.

python scripts/cost.py outputs
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    totals = defaultdict(lambda: [0, 0.0])
    for path in Path(sys.argv[1]).resolve().rglob("wire/*/*/calls.jsonl"):
        model = path.parts[-5]
        for line in path.open():
            usage = (json.loads(line).get("response") or {}).get("usage") or {}
            totals[model][0] += 1
            totals[model][1] += usage.get("cost", 0.0)
    for model, (calls, cost) in sorted(totals.items()):
        print(f"{model:24s} {calls:5d} calls  ${cost:.4f}")
    print(
        f"{'total':24s} {sum(c for c, _ in totals.values()):5d} calls  "
        f"${sum(d for _, d in totals.values()):.4f}"
    )


if __name__ == "__main__":
    main()
