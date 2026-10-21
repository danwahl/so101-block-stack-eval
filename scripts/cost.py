"""What the model calls under a log directory cost, as analysis.py counts it.

python scripts/cost.py outputs
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from analysis import call_usage


def main():
    totals = defaultdict(lambda: [0, 0.0])
    for path in Path(sys.argv[1]).resolve().rglob("wire/*/*/calls.jsonl"):
        model = path.parts[-5]
        for line in path.open():
            totals[model][0] += 1
            totals[model][1] += call_usage(model, json.loads(line))[0]
    for model, (calls, cost) in sorted(totals.items()):
        print(f"{model:24s} {calls:5d} calls  ${cost:.4f}")
    print(
        f"{'total':24s} {sum(c for c, _ in totals.values()):5d} calls  "
        f"${sum(d for _, d in totals.values()):.4f}"
    )


if __name__ == "__main__":
    main()
