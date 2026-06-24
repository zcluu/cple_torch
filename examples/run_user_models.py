import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cple.configs.runner import build_adapter, load_config, write_environment
from cple.runtime.platform import CPLEPlatform
from user_csi_models import build_user_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPLE with user-defined CSI components.")
    parser.add_argument(
        "--config",
        default="configs/user_models.yaml",
        help="YAML config that defines the Sionna scenario and CPLE runtime settings.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = build_adapter(config)
    shape = config.shape.to_spec()
    network = build_user_flow(shape=shape, device=config.platform.device, flow=config.flow)

    platform = CPLEPlatform(config.platform, shape, config.feedback, adapter)
    platform.run(network)
    outputs = platform.export_outputs(config.platform.output_dir)
    output_dir = Path(config.platform.output_dir)
    write_environment(output_dir / "environment.txt")

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"environment: {output_dir / 'environment.txt'}")
