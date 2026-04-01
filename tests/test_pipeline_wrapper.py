import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_all_wrapper_dry_run_completes():
    env = os.environ.copy()
    env.update(
        {
            "SKIP_VENV_CHECK": "1",
            "SKIP_GPU_CHECK": "1",
            "SKIP_DATASET_CHECK": "1",
            "DRY_RUN_COMMANDS": "1",
            "FAST_DRY_RUN": "1",
            "RUN_INTERACTIVE": "0",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_all.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert "PIPELINE FINISHED SUCCESSFULLY" in output
    assert "[DRY-RUN] python -m training.train_vlm" in output
    assert "[DRY-RUN] python -m evaluation.evaluate" in output
