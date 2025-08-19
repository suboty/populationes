import json
from pathlib import Path

import numpy as np


class GnbgError(Exception):
    ...


class GNBG:
    """Load GNBG functions as json-files for testing."""

    def __init__(
            self,
            func_num: int,
            path_to_func: Path | str= Path('..', '_gnbg_functions', 'json')
    ):
        # basic parameters
        self.max_evals = None
        self.acceptance_threshold = None
        self.dimension = None
        self.max_coord = None
        self.min_coord = None
        self.component_number = None
        self.optimum_value = None

        # matrix`s
        self.comp_min_pos = None
        self.rotation_matrix = None
        self.optimum_position = None

        self.func_num = func_num
        self.path_to_func = path_to_func
        self.data = {}
        self.load_parameters()
        self.initialize_basic_parameters()

    def load_parameters(self):
        """Load parameters from json files."""
        with open(
                Path(self.path_to_func, f'f{self.func_num}.json'), 'r'
        ) as f:
            self.data = json.load(f)

    def initialize_basic_parameters(self):
        """Initialize basic parameters."""
        self.max_evals = self.data['max_evals']
        self.acceptance_threshold = self.data['acceptance_threshold']
        self.dimension = self.data['dimension']
        self.min_coord = self.data['min_coordinate']
        self.max_coord = self.data['max_coordinate']
        self.component_number = self.data['o']
        self.optimum_value = self.data['optimum_value']

        try:
            self.comp_min_pos = np.array(
                self.data['component_minimum_position']
            ).reshape(self.component_number, -1)
            self.optimum_position = np.array(
                self.data['optimum_position']
            )
            self.rotation_matrix = np.array(self.data['rotation_matrix'])
        except Exception as e:
            raise GnbgError(f'Error while matrix making, error: {e}')

    def fitness(self, x: np.ndarray) -> float:
        z = x - self.optimum_position
        if hasattr(self, 'rotation_matrix') and self.rotation_matrix.shape == (self.dimension, self.dimension):
            z = self.rotation_matrix @ z
        return np.sum(z ** 2)

    def __str__(self):
        return f"GNBG Function {self.func_num} (D={self.dimension}, comp_num={self.component_number})"


if __name__ == '__main__':
    gnbg = GNBG(
        func_num=16,
        path_to_func=Path('json')
    )
    print(gnbg)
