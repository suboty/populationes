from pathlib import Path

import numpy as np


class GNBG:
    """Load GNBG functions as json-files for testing."""

    def __init__(self, func_num: int):
        self.func_num = func_num
        self.path_to_func = Path('..', '_gnbg_functions')

    def __str__(self):
        return f"GNBG Function {self.func_num} (D={self.dimension})"
