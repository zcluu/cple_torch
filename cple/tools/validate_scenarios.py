from __future__ import annotations

import argparse
from pathlib import Path

from cple.scenario import describe_sionna_mapping, load_sionna_scenario, scenario_to_mock_adapter_config, scenario_to_platform_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CPLE Sionna scenario YAML files.")
    parser.add_argument("paths", nargs="+", help="Scenario YAML files to validate")
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        scenario = load_sionna_scenario(path)
        platform = scenario_to_platform_config(scenario)
        adapter = scenario_to_mock_adapter_config(scenario)
        mapping = describe_sionna_mapping(scenario)
        print(f"[OK] {path}")
        print(f"  name={scenario.name}")
        print(f"  scenario={scenario.topology.scenario}")
        print(f"  num_slots={platform.num_slots}, tti_ms={platform.tti_ms}, deadline_ms={platform.deadline_ms}")
        print(f"  num_ues={adapter.num_ues}, scheduled_per_slot={adapter.scheduled_per_slot}, csi_dim={adapter.csi_dim}")
        print(f"  scheduler={mapping['scheduler']['class']}")


if __name__ == "__main__":
    main()
