"""Run from the project root: python3 run_simulator.py --scenario_type Selective."""
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the information diffusion simulator.")
    parser.add_argument(
        "--scenario_type", "--scenario-type",
        type=str.capitalize,
        choices=("Generic", "Selective", "Generative"),
        default=None,
        help="Override the scenario strategy (otherwise keep the configured value).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Parse before importing simulation dependencies so --help works offline.
    from Melodie import Simulator
    from source.config import config
    from source.data_loader import LLMDataLoader
    from source.model import LLMModel
    from source.scenario import LLMScenario

    class CommandLineSimulator(Simulator):
        def generate_scenarios(self):
            scenarios = super().generate_scenarios()
            # Apply after setup and spreadsheet loading, before model creation.
            if args.scenario_type is not None:
                for scenario in scenarios:
                    scenario.scenario_type = args.scenario_type
                    name = scenario.scenario_name
                    prefix, separator, suffix = name.rpartition("_")
                    if separator and suffix.lower() in {"generic", "selective", "generative"}:
                        scenario.scenario_name = f"{prefix}_{args.scenario_type}"
                    else:
                        scenario.scenario_name = f"{name}_{args.scenario_type}"
            return scenarios

    simulator = CommandLineSimulator(
        config=config,
        scenario_cls=LLMScenario,
        model_cls=LLMModel,
        data_loader_cls=LLMDataLoader,
    )
    simulator.run()


if __name__ == "__main__":
    main()
