import json
import glob

runs = glob.glob("experiments/run_*")

for r in runs:

    metrics_file = f"{r}/metrics.json"

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        print(r, metrics)

    except FileNotFoundError:
        print(f"WARNING: metrics missing for {r}: {metrics_file}")
    except json.JSONDecodeError as exc:
        print(f"WARNING: invalid metrics JSON for {r}: {exc}")