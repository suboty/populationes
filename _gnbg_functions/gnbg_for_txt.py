from pathlib import Path

import numpy as np


class GNBG:
    """Load GNBG functions as txt-files for testing."""

    def __init__(self, func_num: int, verbose: bool = False):
        self.sigma_matrices = None
        self.lambda_ = None
        self.omega = None
        self.mu = None
        self.opt_position = None
        self.optimum_value = None
        self.verbose = verbose
        self.rotation_matrix = None
        self.comp_min_pos = None
        self.max_coord = None
        self.min_coord = None
        self.comp_num = None
        self.dimension = None
        self.acceptance_threshold = None
        self.max_evals = None
        self.func_num = func_num
        self.path_to_func = Path('_gnbg_functions', 'txt')
        self.load_parameters()
        self.prepare_components()

    def load_parameters(self):
        with open(Path(self.path_to_func, f"f{self.func_num}.txt"), "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.max_evals = int(float(lines[0]))
        self.acceptance_threshold = float(lines[1])
        self.dimension = int(float(lines[2]))
        self.comp_num = int(float(lines[3]))
        self.min_coord = float(lines[4])
        self.max_coord = float(lines[5])

        self.comp_min_pos = np.array(
            [float(x) for x in lines[6].split()[:self.dimension]]
        ).reshape(1, -1)

        matrix_start = 7
        while matrix_start < len(lines):
            clean_line = lines[matrix_start].strip()
            if clean_line.startswith(('0 ', '1 ')) or clean_line.startswith((' 0', ' 1')):
                break
            matrix_start += 1

        self.rotation_matrix = np.eye(self.dimension)
        try:
            rot_matrix_lines = lines[matrix_start:matrix_start + self.dimension]
            cleaned_matrix = []
            for line in rot_matrix_lines:
                nums = [float(x) for x in line.strip().split()[:self.dimension]]
                if len(nums) == self.dimension:
                    cleaned_matrix.append(nums)

            if len(cleaned_matrix) == self.dimension:
                self.rotation_matrix = np.array(cleaned_matrix)
        except Exception as e:
            if self.verbose:
                print(e)

        opt_value_line = matrix_start + self.dimension
        while opt_value_line < len(lines) \
            and not lines[opt_value_line].replace('.', '', 1).replace('-', '',1).strip().isdigit():
                opt_value_line += 1
        try:
            self.optimum_value = float(lines[opt_value_line])
        except IndexError:
            opt_value_line = len(lines) - 2
            self.optimum_value = float(lines[opt_value_line])
        self.opt_position = np.array(
            [float(x) for x in lines[opt_value_line + 1].split()[:self.dimension]]
        )

        self.mu = 1.0
        self.omega = 0.0
        self.lambda_ = 0.0

    def prepare_components(self):
        self.sigma_matrices = [np.eye(self.dimension)]

    def fitness(self, x: np.ndarray) -> float:
        z = x - self.opt_position
        if hasattr(self, 'rotation_matrix') and self.rotation_matrix.shape == (self.dimension, self.dimension):
            z = self.rotation_matrix @ z
        ans = np.sum(z ** 2)
        return ans

    def __str__(self):
        return f"GNBG Function {self.func_num} (D={self.dimension})"
