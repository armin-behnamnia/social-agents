from Melodie import Simulator
from config import config
from source.data_loader import LLMDataLoader
from source.model import LLMModel
from source.scenario import LLMScenario
from source.visualizer import LLMVisualizer

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="",
        choices=["stance_aware"],
        help="Simulation method"
    )

    args = parser.parse_args()

    simulator = Simulator(
        config=config,
        model_cls=LLMModel,
        scenario_cls=LLMScenario,
        data_loader_cls=LLMDataLoader,
        visualizer_cls=LLMVisualizer,
        method = argparse
    )

    simulator.run()


if __name__ == "__main__":
    main()
