import os
import subprocess
import sys


ENTRY_FILES = [
    "training/train_vlm.py",
    "evaluation/evaluate.py",
    "evaluation/run_benchmarks.py",
    "inference/generate.py",
    "inference/run_chat.py",
]


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    failures = []

    print("Running entrypoint smoke tests (syntax + import dependencies)...")
    for rel_path in ENTRY_FILES:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", abs_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"[OK] {rel_path}")
        except Exception as exc:
            failures.append((rel_path, str(exc)))
            print(f"[FAIL] {rel_path}: {exc}")

    if failures:
        print("\nSmoke test failed. Import errors detected:")
        for rel_path, error in failures:
            print(f"- {rel_path}: {error}")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
