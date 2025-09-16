import numpy as np
from pathlib import Path
from typing import Union

class GNBG:
    def __init__(
            self,
            func_num: int,
            directory: Union[str, Path] = Path('gnbg24', 'txt')
    ):
        # Читаем весь файл как список чисел
        self.func_num = func_num
        path = Path(directory) / f"f{func_num}.txt"
        with open(path, "r") as fin:
            tokens = fin.read().split()
        it = iter(tokens)

        # Метаданные и размеры
        self.fe_eval = 0
        self.acceptance_reach_point = -1.0

        self.max_evals = int(next(it))
        self.acceptance_threshold = float(next(it))
        self.dimension = int(next(it))
        self.comp_num = int(next(it))
        self.min_coord = float(next(it))
        self.max_coord = float(next(it))

        D, C = self.dimension, self.comp_num

        # Массивы
        self.fe_history = np.zeros(self.max_evals+1, dtype=float)
        self.a = np.zeros(D, dtype=float)
        self.temp = np.zeros(D, dtype=float)
        self.opt_position = np.zeros(D, dtype=float)
        self.comp_sigma = np.zeros(C, dtype=float)
        self.lambda_ = np.zeros(C, dtype=float)
        self.comp_min_pos = np.zeros((C, D), dtype=float)
        self.comp_h = np.zeros((C, D), dtype=float)
        self.mu = np.zeros((C, 2), dtype=float)
        self.omega = np.zeros((C, 4), dtype=float)
        self.rotation_matrix  = np.zeros((C, D, D), dtype=float)

        # Загрузка параметров
        for i in range(C):
            for j in range(D):
                self.comp_min_pos[i, j] = float(next(it))
        for i in range(C):
            self.comp_sigma[i] = float(next(it))
        for i in range(C):
            for j in range(D):
                self.comp_h[i, j] = float(next(it))
        for i in range(C):
            for j in range(2):
                self.mu[i, j] = float(next(it))
        for i in range(C):
            for j in range(4):
                self.omega[i, j] = float(next(it))
        for i in range(C):
            self.lambda_[i] = float(next(it))
        for j in range(D):
            for k in range(D):
                for i in range(C):
                    self.rotation_matrix[i, j, k] = float(next(it))
        self.optimum_value = float(next(it))
        for j in range(D):
            self.opt_position[j] = float(next(it))

        self.best_found_result = float("inf")
        self.f_val = 0.0

    def fitness(self, xvec: np.ndarray) -> float:
        xvec = np.asarray(xvec, dtype=float)
        assert xvec.shape == (self.dimension,), "xvec должен иметь форму (dimension,)"

        res = None
        for i in range(self.comp_num):
            a = xvec - self.comp_min_pos[i]

            temp = self.rotation_matrix[i] @ a

            a_new = np.zeros_like(temp)
            pos = temp > 0
            neg = temp < 0
            if np.any(pos):
                tp = temp[pos]
                ln_tp = np.log(tp)
                a_new[pos] = np.exp(
                    ln_tp
                    + self.mu[i, 0] * (
                        np.sin(self.omega[i, 0] * ln_tp)
                        + np.sin(self.omega[i, 1] * ln_tp)
                    )
                )
            if np.any(neg):
                tn = -temp[neg]
                ln_tn = np.log(tn)
                a_new[neg] = -np.exp(
                    ln_tn
                    + self.mu[i, 1] * (
                        np.sin(self.omega[i, 2] * ln_tn)
                        + np.sin(self.omega[i, 3] * ln_tn)
                    )
                )

            f_val_inner = np.sum((a_new * a_new) * self.comp_h[i])
            f_val = self.comp_sigma[i] + (f_val_inner ** self.lambda_[i])

            res = f_val if (res is None) else min(res, f_val)

        if self.fe_eval > self.max_evals:
            return float(res)

        self.fe_history[self.fe_eval] = res
        if self.fe_eval == 0:
            self.best_found_result = res
        else:
            self.best_found_result = min(res, self.best_found_result)

        try:
            if (self.fe_history[self.fe_eval] - self.optimum_value < self.acceptance_threshold
                and self.acceptance_reach_point == -1):
                self.acceptance_reach_point = float(self.fe_eval)
        except Exception as e:
            raise e

        self.fe_eval += 1
        return float(res)

    def __str__(self):
        return f"GNBG Function {self.func_num} (D={self.dimension})"
