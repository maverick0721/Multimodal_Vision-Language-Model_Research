import json
import glob

runs = glob.glob("experiments/run_*")

for r in runs:

    metrics_file = f"{r}/metrics.json"

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        print(r, metrics)

    except:
        pass