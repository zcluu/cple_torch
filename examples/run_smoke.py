from cple import run_experiment


if __name__ == "__main__":
    outputs = run_experiment("configs/smoke.yaml")
    for name, path in outputs.items():
        print(f"{name}: {path}")
