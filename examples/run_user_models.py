import argparse

from cple import CPLEPlatform
from cple.runner import build_adapter, load_config, write_environment
from user_csi_models import build_user_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPLE with user-defined PyTorch CSI models.")
    parser.add_argument(
        "--config",
        default="configs/user_models.yaml",
        help="YAML config that defines the Sionna scenario and CPLE runtime settings.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = build_adapter(config)
    models = build_user_models(csi_dim=config.adapter.csi_dim, horizon=3)

    platform = CPLEPlatform(config.platform, adapter, models)
    platform.run()
    outputs = platform.export_outputs(config.platform.output_dir)
    from pathlib import Path

    output_dir = Path(config.platform.output_dir)
    write_environment(output_dir / "environment.txt")

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"environment: {output_dir / 'environment.txt'}")
