from __future__ import annotations

import argparse
from pathlib import Path

from cple.data.scenario import (
    describe_sionna_mapping,
    load_sionna_scenario,
    scenario_to_adapter_config,
    scenario_to_feedback_config,
    scenario_to_platform_config,
    scenario_to_shape_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CPLE Sionna scenario YAML files.")
    parser.add_argument("paths", nargs="+", help="Scenario YAML files to validate")
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        scenario = load_sionna_scenario(path)
        platform = scenario_to_platform_config(scenario)
        adapter = scenario_to_adapter_config(scenario)
        shape = scenario_to_shape_config(scenario)
        feedback = scenario_to_feedback_config(scenario)
        mapping = describe_sionna_mapping(scenario)
        print(f"[OK] {path}")
        print(f"  name={scenario.name}")
        print(f"  scenario={scenario.topology.scenario}")
        print(f"  num_slots={platform.num_slots}, tti_ms={platform.tti_ms}, deadline_ms={platform.deadline_ms}")
        print(
            "  "
            f"num_ues={adapter.num_ues}, scheduled_per_slot={adapter.scheduled_per_slot}, "
            f"frame_shape={shape.frame_shape}, dtype={shape.dtype}, "
            f"history_len={shape.history_len}, horizon={shape.horizon}"
        )
        print(
            "  "
            f"feedback_bitwidth={feedback.bitwidth}, "
            f"bits_per_resource_unit={feedback.bits_per_resource_unit}"
        )
        print(f"  scheduler={mapping['scheduler']['class']}")


if __name__ == "__main__":
    main()
