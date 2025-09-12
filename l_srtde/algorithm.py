import time
import random
from typing import List, Callable, Union

import numpy as np
import numpy.typing as npt


class AlgorithmConst:
    """
    Константы для алгоритма.

    Аттрибуты:
        records_number_per_function:
            количество точек данных, сохраненных в массиве результатов, для построения кривой сходимости.
        global_seed: глобальный seed для всех генераторов.
        seed1, seed2: seed-ы для генераторов.
    """
    records_number_per_function = 1001
    global_seed = 2025
    seed1 = global_seed
    seed2 = global_seed + 100


class AlgorithmGlobals:
    """
    Глобальные переменные для алгоритма.

    Аттрибуты:
        eval_steps:
            массив контрольных значений оценки,
            в котором будет храниться ошибка оптимизации.
        result_array: хранит значения оптимума.
        sr_array: хранит значения истории успешности.
        last_eval_step: счетчик для хранения последнего шага оценки.
        eval_func_calls: счетчик для хранения количества текущих вызовов целевой функции.
        max_eval_func_calls: максимальное количество вызовов целевой функции (ограничение).
        problem_dimension: размерность задачи.
        eval_func_opt_value: оптимальное значение целевой функции.
        global_best: наилучшее значение целевой функции, найденное глобально (в текущей итерации).
        global_best_init: булев флаг для инициализации global_best.
    """
    eval_steps = np.zeros(AlgorithmConst.records_number_per_function - 1)
    result_array = np.zeros(AlgorithmConst.records_number_per_function)
    sr_array = np.zeros(AlgorithmConst.records_number_per_function)
    last_eval_step = 0
    eval_func_calls = 0
    max_eval_func_calls = 0
    problem_dimension = 30
    eval_func_opt_value = None
    global_best = np.inf
    global_best_init = False


class AlgorithmRandomGenerators:
    """Класс, реализующий генераторы случайных чисел."""

    def __init__(self, seeds: List[int]):
        self.generators = []
        if len(seeds) < 2:
            raise ValueError
        for i, seed in enumerate(seeds):
            self.generators.append(np.random.default_rng(seed=seed))

    def random_integers(self, low: int, high: int, size: int = 1):
        """Возвращает случайное целое число в указанном интервале."""
        return self.generators[0].integers(low=low, high=high, size=size)

    def random_floats(self):
        """Возвращает число с плавающей запятой в интервале от 0 до 1."""
        return self.generators[1].random(size=1)[0]


def quick_sort_with_indices(
        array: Union[npt.NDArray[np.float64], List[float]],
        left: int = 0,
        right: int = None,
        indices: Union[npt.NDArray[np.float64], List[float]] = None
):
    """Быстрая сортировка на месте с соответствующим отслеживанием индекса."""
    if right is None:
        right = len(array) - 1
    if indices is None:
        indices = np.arange(len(array))

    if len(array) == 0 or right < left:
        return array, indices

    i = left
    j = right
    pivot = float(array[(left + right) // 2])

    while i <= j:
        while array[i] < pivot:
            i += 1
        while array[j] > pivot:
            j -= 1
        if i <= j:
            array[i], array[j] = array[j], array[i]
            indices[i], indices[j] = indices[j], indices[i]
            i += 1
            j -= 1

    if left < j:
        quick_sort_with_indices(array, left, j, indices)
    if i < right:
        quick_sort_with_indices(array, i, right, indices)

    return array, indices


class Algorithm:
    consts = AlgorithmConst()
    global_variables = AlgorithmGlobals()
    _random_generators = AlgorithmRandomGenerators(seeds=[
        consts.seed1,
        consts.seed2,
    ])

    threshold = 1e-6

    """Python-имплементация L-STRDE алгоритма."""

    def __init__(
            self,
            fitness_function: Callable,
            problem_dimension: int,
            population_size: int,
            left: int = -100,
            right: int = 100,
            verbose: bool = True
    ):
        self.is_last_call = False
        self.t0 = time.time()
        self.verbose = verbose

        # Инициализация целевой функции
        self.fitness_function = fitness_function

        # Счетчики и локальные переменные
        self.memory_size = 5
        self.memory_iter = 0
        self.memory_index = 0
        self.success_filled = 0

        # Обнуление глобальных счетчиков
        self.global_variables.eval_func_calls = 0
        self.global_variables.last_eval_step = 0
        self.global_variables.global_best = np.inf
        self.global_variables.global_best_init = False

        # Размерность пространства решений
        self.n_vars = problem_dimension

        # Размеры популяции
        self.n_inds_current = population_size
        self.n_inds_front = population_size
        self.n_inds_front_max = population_size
        self.new_n_inds_front = 0
        self.population_size = population_size * 2

        # Индексы и счетчики
        self.chosen_index = 0
        self.iter_number = 0
        self.pf_index = 0

        # Наилучшее значение целевой функции
        self.best_fitness_value = np.inf
        # Доля успешных мутаций
        self.success_rate = 0.5
        # Параметры ДЭ
        self.f = 0.0
        self.cr = 0.0
        # Границы решения
        self.left, self.right = left, right

        # Списки
        self.population = []
        self.front_population = []
        self.temp_population = []
        self.fitness_func_values = []
        self.fitness_func_values_copy = []
        self.fitness_func_values_front = []
        self.trial_solution = []
        self.temp_success_cr = []
        self.cr_memory = []
        self.fitness_values_dif = []
        self.weights = []
        self.indices = []
        self.indices2 = []

        # Подготовка шагов оценки
        self.global_variables.eval_steps = [
            10000.0 / (self.consts.records_number_per_function - 1) * self.n_vars * (steps_k + 1)
            for steps_k in range(self.consts.records_number_per_function - 1)
        ]

        # Инициализация массивов
        self.population = np.random.uniform(self.left, self.right, size=(self.population_size, self.n_vars))
        self.front_population = np.zeros((self.n_inds_front, self.n_vars))
        self.temp_population = np.zeros((self.population_size, self.n_vars))

        self.fitness_func_values = np.zeros(self.population_size)
        self.fitness_func_values_copy = np.zeros(self.population_size)
        self.fitness_func_values_front = np.zeros(self.n_inds_front)

        self.weights = np.zeros(self.population_size)
        self.temp_success_cr = np.zeros(self.population_size)
        self.fitness_values_dif = np.zeros(self.population_size)

        self.cr_memory = np.ones(self.memory_size)
        self.trial_solution = np.zeros(self.n_vars)

        self.indices = np.zeros(self.population_size, dtype=int)
        self.indices2 = np.zeros(self.population_size, dtype=int)

    def update_cr_memory(self):
        """
        Функция отвечает за адаптацию параметра кроссовера (Cr)
        на основе успешных значений, найденных в текущем поколении.
        """
        if self.success_filled != 0:
            new_cr = 0.5 * ( self.mean_wl(
                self.temp_success_cr, self.fitness_values_dif
            ) + self.cr_memory[self.memory_iter])
            self.cr_memory[self.memory_iter] = new_cr
            self.memory_iter = (self.memory_iter + 1) % self.memory_size

    def mean_wl(
            self,
            array: Union[npt.NDArray[np.float64], List[float]],
            temp_weights: Union[npt.NDArray[np.float64], List[float]],
    ) -> float:
        """
        Функция возвращает адаптивное значение параметра,
        основанное на успешных значениях из предыдущего поколения,
        с учетом того, насколько они улучшили результат.
        """
        sum_weight = np.sum(temp_weights)
        if sum_weight == 0:
            return 1.0

        weights = temp_weights / sum_weight
        sum_square = np.sum(weights * array * array)
        sum_val = np.sum(weights * array)

        if abs(sum_val) > self.threshold:
            return np.divide(sum_square, sum_val)
        else:
            return 1.0

    def find_n_save_best(self, init, ind_iter):
        """
        Функция обновляет локальные
        и глобальные лучшие значения целевой функции.
        """
        if self.fitness_func_values[ind_iter] <= self.best_fitness_value or init:
            self.best_fitness_value = self.fitness_func_values[ind_iter]
        if self.best_fitness_value < self.global_variables.global_best or init:
            self.global_variables.global_best = self.best_fitness_value

    def save_best_values(self):
        """
        Функция предназначена для сохранения сходимости
        траектории алгоритма оптимизации.
        """
        temp = self.global_variables.global_best - self.global_variables.eval_func_opt_value
        if temp <= self.threshold and self.global_variables.result_array[
            self.consts.records_number_per_function - 1
        ] == self.global_variables.max_eval_func_calls:
            self.global_variables.result_array[
                self.consts.records_number_per_function - 1
                ] = self.global_variables.eval_func_calls

        for step_eval_func_count in range(
                self.global_variables.last_eval_step,
                self.consts.records_number_per_function - 1
        ):
            if self.global_variables.eval_func_calls == self.global_variables.eval_steps[
                step_eval_func_count
            ]:
                if temp <= self.threshold:
                    temp = 0

                # Обновление результирующих массивов
                self.global_variables.result_array[step_eval_func_count] = temp
                self.global_variables.sr_array[step_eval_func_count] = self.success_rate

                self.global_variables.last_eval_step = step_eval_func_count

    @staticmethod
    def reflect(val, left, right):
        if val < left:
            return left + (left - val)
        elif val > right:
            return right - (val - right)
        return val

    def remove_worst(self, _n_inds_front: int, new_n_inds_front: int):
        """
        Удаление худших особей из популяции
        по значению целевой функции.
        """
        points_to_remove = _n_inds_front - new_n_inds_front

        for l in range(points_to_remove):
            worst = np.argmax(self.fitness_func_values_front[:_n_inds_front])

            # Сдвиг
            self.front_population[
                worst:_n_inds_front - 1
            ] = self.front_population[worst + 1:_n_inds_front]
            self.fitness_func_values_front[
                worst:_n_inds_front - 1
            ] = self.fitness_func_values_front[worst + 1:_n_inds_front]
            _n_inds_front -= 1

            # «Очистить» хвост
            self.front_population[new_n_inds_front:] = 0.0
            self.fitness_func_values_front[new_n_inds_front:] = np.inf
        return _n_inds_front

    @staticmethod
    def create_component_selector(fit_temp2):
        """
        Функция создает функцию-генератор, которая возвращает
        один индекс, выбранный случайным образом с учетом весов.
        """

        def selector():
            return random.choices(range(len(fit_temp2)), weights=fit_temp2, k=1)[0]

        return selector

    @staticmethod
    def minmax(array, limit):
        min_value = np.min(array[:limit])
        max_value = np.max(array[:limit])
        return min_value, max_value

    def __call__(self):
        """Запуск алгоритма L-SRTDE."""
        if self.verbose:
            print(f"Starting main cycle: max evals {self.global_variables.max_eval_func_calls}")

        # Инициализация значений целевой функции для стартовой популяции
        for i_inds in range(self.n_inds_front):
            self.fitness_func_values[i_inds] = self.fitness_function(self.population[i_inds])
            self.global_variables.eval_func_calls += 1
            self.find_n_save_best(i_inds == 0, i_inds)

            if (not self.global_variables.global_best_init
                    or self.best_fitness_value < self.global_variables.global_best):
                self.global_variables.global_best = self.best_fitness_value
                self.global_variables.global_best_init = True

            self.save_best_values()

        # Копирование значений целевых функций и индексов
        for i in range(self.n_inds_front):
            self.fitness_func_values_copy[i] = self.fitness_func_values[i].copy()
            self.indices[i] = i

        # Определение мин/макс целевой функции        
        min_fitness_value, max_fitness_value = self.minmax(
            array=self.fitness_func_values,
            limit=self.n_inds_front
        )

        # Сортировка и обновление фронтовой части популяции
        if min_fitness_value != max_fitness_value:
            self.fitness_func_values_copy, self.indices = quick_sort_with_indices(
                array=self.fitness_func_values_copy,
                indices=self.indices,
                left=0,
                right=self.n_inds_front - 1,
            )

        # Обновляем фронтовую часть популяции согласно сортировке
        for i in range(self.n_inds_front):
            self.front_population[i] = self.population[self.indices[i]].copy()
            self.fitness_func_values_front[i] = self.fitness_func_values_copy[i]

        self.pf_index = 0
        epoch_num = 0

        # Главный цикл алгоритма
        # Пока не дойдем до максимума вычислений целевой функции
        while self.global_variables.eval_func_calls < self.global_variables.max_eval_func_calls:
            epoch_num += 1

            _dif_calls = self.global_variables.max_eval_func_calls - self.global_variables.eval_func_calls
            if self.verbose and _dif_calls < 50_000 and not self.is_last_call:
                print(f'### Left {_dif_calls} calls')
                self.is_last_call = True

            _max_calls = self.global_variables.max_eval_func_calls
            _current_calls = self.global_variables.eval_func_calls

            if self.verbose and self.iter_number % (self.global_variables.max_eval_func_calls // 1_000) == 0:
                print(
                    f'--- Iteration {self.iter_number}, '
                    f'left {_max_calls-_current_calls} calls, '
                    f'current best fitness value: {self.global_variables.global_best}, '
                    f'elapsed: {round(time.time()-self.t0, 2)} seconds'
                )
                self.t0 = time.time()

            # Расчет параметров
            mean_F = 0.4 + np.tanh(self.success_rate * 5) * 0.25
            sigma_F = 0.02

            # Копируем и сортируем значения целевой функции для всей популяции
            for i in range(self.n_inds_front):
                self.fitness_func_values_copy[i] = self.fitness_func_values[i]
                self.indices[i] = i

            min_fitness_value, max_fitness_value = self.minmax(
                array=self.fitness_func_values,
                limit=self.n_inds_front
            )

            if min_fitness_value != max_fitness_value:
                self.fitness_func_values_copy, self.indices = quick_sort_with_indices(
                    array=self.fitness_func_values_copy,
                    indices=self.indices,
                    left=0,
                    right=self.n_inds_front - 1,
                )

            # Аналогично сортируем значения целевой функции для фронтовой части популяции
            for i in range(self.n_inds_front):
                self.fitness_func_values_copy[i] = self.fitness_func_values_front[i]
                self.indices2[i] = i

            min_fitness_value, max_fitness_value = self.minmax(
                array=self.fitness_func_values_front,
                limit=self.n_inds_front
            )

            if min_fitness_value != max_fitness_value:
                self.fitness_func_values_copy, self.indices2 = quick_sort_with_indices(
                    array=self.fitness_func_values_copy,
                    indices=self.indices2,
                    left=0,
                    right=self.n_inds_front - 1,
                )

            # Весовой вектор для выбора
            fitness_temp2 = np.empty(self.n_inds_front)
            for i in range(self.n_inds_front):
                fitness_temp2[i] = np.exp(-i / self.n_inds_front * 3)
            component_selector_front = self.create_component_selector(fitness_temp2)

            p_size_val = max(
                2,
                int(self.n_inds_front * 0.7 * np.exp(-self.success_rate * 7))
            )

            # Итерируемся по фронтовой части популяции
            for i_ind in range(self.n_inds_front):

                verbose_i = self._random_generators.random_integers(low=0, high=self.n_inds_front, size=3)

                # Выбираем индекс из фронтовой части популяции
                self.chosen_index = self._random_generators.random_integers(
                    low=0, high=self.n_inds_front - 1
                )
                # Выбираем индекс выбранного индивида в координатах population
                chosen_pop_idx = self.indices2[self.chosen_index]

                # Выбираем индекс из памяти
                self.memory_index = self._random_generators.random_integers(
                    low=0, high=self.memory_size - 1
                )

                # Выбор p_rand, не равного индексу выбранного индивида
                p_rand = self.indices[
                    self._random_generators.random_integers(
                        low=0, high=p_size_val - 1
                    )
                ]
                max_at, at = 100, 0
                while p_rand == chosen_pop_idx and max_at < at:
                    p_rand = self.indices[
                        self._random_generators.random_integers(
                            low=0, high=p_size_val - 1
                        )
                    ]
                    at += 1
                if at == max_at:
                    p_rand = self.indices[
                        self._random_generators.random_integers(
                            low=0, high=p_size_val - 1
                        )
                    ]

                # Выбор rand1, не равного p_rand
                rand1 = self.indices2[
                    self._random_generators.random_integers(
                        low=0, high=self.n_inds_front - 1
                    )
                ]
                rand1_pop = self.indices2[rand1]
                max_at, at = 100, 0
                while (rand1_pop == p_rand or rand1_pop == chosen_pop_idx) and max_at < at:
                    rand1 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=self.n_inds_front - 1
                        )
                    ]
                    rand1_pop = self.indices2[rand1]
                    at += 1
                if at == max_at:
                    rand1 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=p_size_val - 1
                        )
                    ]

                # Выбор rand2, не равного p_rand и rand1
                rand2 = self.indices2[component_selector_front()]
                rand2_pop = self.indices2[rand2]
                max_at, at = 100, 0
                while (rand2_pop == p_rand or rand2_pop == rand1_pop
                       or rand2_pop == chosen_pop_idx) and max_at < at:
                    rand2 = self.indices2[component_selector_front()]
                    rand2_pop = self.indices2[rand2]
                    at += 1
                if max_at == at:
                    rand2 = self.indices2[component_selector_front()]

                # Выбор rand2, не равного p_rand, rand1 и rand2
                rand3 = self.indices2[
                    self._random_generators.random_integers(
                        low=0, high=self.n_inds_front - 1
                    )
                ]
                max_at, at = 100, 0
                while (rand3 == p_rand or rand3 == rand1_pop
                       or rand3 == rand2_pop or rand3 == chosen_pop_idx) and max_at < at:
                    rand3 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=self.n_inds_front - 1
                        )
                    ]
                    at += 1
                if at == max_at:
                    rand3 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=self.n_inds_front - 1
                        )
                    ]

                # Генерация F и cr с ограничениями
                # Параметр мутации F
                self.f = np.clip(np.random.normal(mean_F, sigma_F), 0.0, 1.0)

                # Параметр кроссовера cr из памяти
                self.cr = np.random.normal(self.cr_memory[self.memory_index], 0.05)
                # Ограничиваем cr диапазоном [0, 1]
                self.cr = min(max(self.cr, 0.0), 1.0)
                if type(self.cr) == np.ndarray:
                    self.cr = self.cr[0]
                actual_cr = 0
                will_crossover = random.randint(0, self.n_vars - 1)

                # Создаем массив для пробного решения
                self.trial_solution = np.zeros(self.n_vars)

                for j in range(self.n_vars):
                    if self._random_generators.random_floats() < self.cr or will_crossover == j:
                        val = (self.population[rand1].reshape(-1)[j]
                               + self.f * (self.front_population[p_rand].reshape(-1)[j]
                                           - self.population[self.chosen_index].reshape(-1)[j])
                               + self.f * (self.population[rand2].reshape(-1)[j]
                                           - self.front_population[rand3].reshape(-1)[j]))

                        val = self.reflect(val, self.left, self.right)
                        self.trial_solution[j] = val
                        actual_cr += 1
                    else:
                        self.trial_solution[j] = self.front_population[self.chosen_index].reshape(-1)[j]

                actual_cr /= float(self.n_vars)

                # Считаем значение целевой функции для пробного решения
                temp_fit = self.fitness_function(self.trial_solution)
                self.global_variables.eval_func_calls += 1

                if temp_fit <= self.fitness_func_values_front[self.chosen_index]:
                    # Если улучшение найдено
                    idx = self.n_inds_front + self.success_filled
                    self.population[idx] = self.trial_solution.copy()
                    self.front_population[self.pf_index] = self.trial_solution.copy()
                    self.fitness_func_values[idx] = temp_fit
                    self.fitness_func_values_front[self.pf_index] = temp_fit
                    # Обновляем лучшее решение
                    self.find_n_save_best(False, idx)
                    # Сохраняем cr
                    self.temp_success_cr[self.success_filled] = actual_cr
                    self.fitness_values_dif[self.success_filled] = abs(
                        self.fitness_func_values_front[self.chosen_index] - temp_fit
                    )

                    self.success_filled += 1
                    # Сдвигаем индекс pf_index
                    self.pf_index = (self.pf_index + 1) % self.n_inds_front

                self.save_best_values()

                temp_fit_val = float(temp_fit) if np.isscalar(temp_fit) else float(temp_fit.item())
                front_val = float(self.fitness_func_values_front[self.chosen_index])

                if (self.verbose and temp_fit <= self.fitness_func_values_front[self.chosen_index]
                        and i_ind in verbose_i):
                    print(f"\t Success: new fitness={temp_fit_val:.4f}, "
                          f"improvement={front_val - temp_fit_val:.4f}, "
                          f"pf_index={self.pf_index}")

            # Вычисляем процент успешных мутаций
            self.success_rate = float(self.success_filled) / float(self.n_inds_front)

            if self.n_inds_front < 4:
                raise ValueError(f"Слишком мало индивидов в фронтовой популяции: {self.n_inds_front}")

            # Вычисляем новый размер фронтовой части популяции
            self.new_n_inds_front = int(
                (4 - self.n_inds_front_max) / self.global_variables.max_eval_func_calls
                * self.global_variables.eval_func_calls
                + self.n_inds_front_max
            )
            self.new_n_inds_front = max(4, self.new_n_inds_front)

            # Удаляем худшие решения из фронтовой части популяции
            self.remove_worst(self.n_inds_front, self.new_n_inds_front)
            # Обновляем размер фронтовой части популяции
            self.n_inds_front = self.new_n_inds_front
            # Обновляем память cr
            self.update_cr_memory()
            # Обновляем текущий размер популяции
            self.n_inds_current = self.n_inds_front + self.success_filled
            if epoch_num % 1000 == 0 and epoch_num > 0 and self.verbose:
                print(f'\tEpoch {epoch_num} | eval calls: {self.global_variables.eval_func_calls} '
                      f'| Current Optimum: {round(self.global_variables.global_best,3)} '
                      f'| F: {round(mean_F,2)} | Cr: {round(self.cr,2)} | Front size: {self.n_inds_front} '
                      f'| SR: {round(self.success_rate,3)} | SF: {self.success_filled}')
            # Обнуляем количество успешных мутаций
            self.success_filled = 0
            # Увеличиваем счетчик поколений
            self.iter_number += 1

            # Коррекция популяции, если она увеличилась
            if self.n_inds_current > self.n_inds_front:
                for i in range(self.n_inds_current):
                    self.indices[i] = i

                min_fitness_value = np.min(self.fitness_func_values[:self.n_inds_current])
                max_fitness_value = np.max(self.fitness_func_values[:self.n_inds_current])

                if min_fitness_value != max_fitness_value:
                    self.fitness_func_values, self.indices = quick_sort_with_indices(
                        array=self.fitness_func_values,
                        indices=self.indices,
                        left=0,
                        right=self.n_inds_current - 1,
                    )

                self.n_inds_current = self.n_inds_front

                for i in range(self.n_inds_current):
                    self.temp_population[i] = self.population[self.indices[i]].copy()
                    self.population[i] = self.temp_population[i].copy()

            if self.verbose and self.iter_number % (self.global_variables.max_eval_func_calls // 1_000) == 0:
                print(f"End of iteration {self.iter_number}: global_best={self.global_variables.global_best}, "
                      f"n_inds_front={self.n_inds_front}, success_rate={self.success_rate:.3f}")

        if self.verbose:
            print(f'--- Global best: {self.global_variables.global_best}')

        return None, self.global_variables.global_best
