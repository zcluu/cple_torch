import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cple.configs.runner import run_experiment


if __name__ == "__main__":
    outputs = run_experiment("configs/smoke.yaml")
    for name, path in outputs.items():
        print(f"{name}: {path}")
