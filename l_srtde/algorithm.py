import random
from typing import List, Callable, Any

import numpy as np


class AlgorithmConst:
    """
    Константы для алгоритма.

    Аттрибуты:
        records_number_per_function:
            количество точек данных, сохраненных в массиве результатов, для построения кривой сходимости.
        global_seed: глобальный seed для всех генераторов.
        seed1,...,seed4: seed-ы для генераторов.
    """
    records_number_per_function = 1001
    global_seed = 2025
    seed1 = global_seed
    seed2 = global_seed + 100
    seed3 = global_seed + 200
    seed4 = global_seed + 300


class AlgorithmGLobals:
    """
    Глобальные переменные для алгоритма.

    Аттрибуты:
        eval_steps:
            массив контрольных значений оценки,
            в котором будет храниться ошибка оптимизации.
        results_array: хранит значения отклонения от оптимума (ошибки).
        best_of_best_array: хранит текущие значения оптимума.
        last_eval_step: счетчик для хранения последнего шага оценки.
        eval_func_calls: счетчик для хранения количества текущих вызовов целевой функции.
        max_eval_func_calls: максимальное количество вызовов целевой функции (ограничение).
        problem_dimension: размерность задачи.
        eval_func_opt_value: оптимальное значение целевой функции.
        global_best: наилучшее значение целевой функции, найденное глобально (в текущей итерации).
        global_best_init: булев флаг для инициализации global_best.
    """
    eval_steps = np.zeros(AlgorithmConst.records_number_per_function - 1)
    results_array = np.zeros(AlgorithmConst.records_number_per_function)
    best_of_best_array = np.zeros(AlgorithmConst.records_number_per_function)
    last_eval_step = 0
    eval_func_calls = 0

    # Шаг 1: Инициализация максимального количество вызовов целевой функции
    max_eval_func_calls = 0
    problem_dimension = 30
    eval_func_opt_value = 0.0
    global_best = np.inf
    global_best_init = False


class AlgorithmRandomGenerators:
    """Класс, реализующий генераторы."""

    def __init__(self, seeds: List[int]):
        self.generators = []
        if len(seeds) < 4:
            raise ValueError
        for i, seed in enumerate(seeds):
            self.generators.append(np.random.default_rng(seed=seed))

    def random_integers(self, low: int, high: int, size: int = 1):
        """Возвращает случайное целое число в указанном интервале."""
        return self.generators[0].integers(low=low, high=high, size=size)

    def random_floats(self):
        """Возвращает число с плавающей запятой в интервале от 0 до 1."""
        return self.generators[1].random(size=1)[0]

    def random_normal(self, size: int = 1):
        """Возращает случайное число с плавающей запятой в нормальном распределении."""
        return self.generators[2].normal(size=size)

    def alt_random_integers(self, low: int, high: int, size: int):
        """Возвращает случайное целое число в указанном интервале. Ещё один."""
        return self.generators[3].integers(low=low, high=high, size=size)


random_generator = AlgorithmRandomGenerators(seeds=[
    AlgorithmConst.seed1,
    AlgorithmConst.seed2,
    AlgorithmConst.seed3,
    AlgorithmConst.seed4
])


def quick_sort_with_indices(
        array: np.ndarray | List[Any],
        left: int = 0, right:
        int = None,
        indices: np.ndarray | List[Any] = None
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
    _consts = AlgorithmConst()
    _global_variables = AlgorithmGLobals()
    _random_generators = AlgorithmRandomGenerators(seeds=[
        _consts.seed1,
        _consts.seed2,
        _consts.seed3,
        _consts.seed4
    ])

    """Python-имплементация L-STRDE алгоритма."""

    def __init__(self, fitness_function: Callable, verbose: bool = True):
        self.verbose = verbose

        # Шаг 1: Инициализация целевой функции
        # target function f(x)
        self.fitness_function = fitness_function

        # Счетчики и локальные переменные

        # Память для параметров ДЭ
        self.memory_size = 5
        self.memory_iter = 0
        self.memory_index = 0
        # Количество успешных решений в итерации
        self.success_filled = 0
        # Размерность пространства решений
        self.n_vars = 0
        # Размер текущей популяции
        self.n_inds_current = 0
        # Размер фронтовой части популяции
        self.n_inds_front = 0
        # Максимальный размер фронтовой части популяции
        self.n_inds_front_max = 0
        # Новый размер фронтовой части популяции
        self.new_n_inds_front = 0
        # Размер популяции
        self.population_size = 0
        # Индекс индивида, выбранного для мутации
        self.chosen_index = 0
        # Номер текущего поколения
        self.iter_number = 0
        # Индекс в фронтовой части, куда записывается новая успешная особь.
        self.pf_index = 0
        # Наилучшее значение целевой функции
        self.best_fitness_value = np.inf
        # Доля успешных мутаций
        self.success_rate = 0.5
        # Параметры ДЭ
        self.F = 0.0
        self.Cr = 0.0
        # Верхняя граница для решения
        self.right = 0.0
        # Нижняя граница для решения
        self.left = 0.0

        # Списки
        self.population = []
        self.front_population = []
        self.temp_population = []
        self.fitness_func_values = []
        self.fitness_func_values_copy = []
        self.fitness_func_values_front = []
        # Пробное решение (индивид)
        self.trial_solution = []
        # Значения параметра Cr для успешных пробных особей в текущей итерации
        self.temp_success_Cr = []
        # Исторические значения Cr — память об успешных итерациях
        self.Cr_memory = []
        # Разница между старыми и новыми значениями целевой функции
        self.fitness_values_dif = []
        # Используются как веса для адаптации Cr
        self.weights = []

        # Индексы популяций
        self.indices = []
        self.indices2 = []

    def initialize(
            self,
            new_n_inds: int,
            new_n_vars: int,
    ):
        """Инициализация всех структур и параметров, необходимых для запуска алгоритма."""

        # Шаг 1: Инициализация размерности пространства решений
        self.n_vars = new_n_vars

        # Шаг 1: Инициализация размера популяции
        self.n_inds_current = new_n_inds

        # Шаг 3: Инициализация размера популяции в фронтовой части
        self.n_inds_front = new_n_inds
        self.n_inds_front_max = new_n_inds
        self.population_size = new_n_inds * 2
        self.left = -100
        self.right = 100
        self.iter_number = 0
        self.chosen_index = 0

        # Шаг 3: Инициализация памяти для параметров
        self.memory_size = 5
        self.memory_iter = 0
        self.success_filled = 0

        # Шаг 4: Показатель успешности ставим в 50%
        self.success_rate = 0.5

        # Подготовка шагов оценки
        self._global_variables.eval_steps = [
            10000.0 / (
                    self._consts.records_number_per_function - 1
            ) * self.n_vars * (steps_k + 1)
            for steps_k in range(self._consts.records_number_per_function - 1)
        ]

        # Инициализация списков
        # Популяция
        self.population = np.zeros((self.population_size, self.n_vars))
        self.front_population = np.zeros((self.n_inds_front, self.n_vars))
        self.temp_population = np.zeros((self.population_size, self.n_vars))
        # Значения целевой функции
        self.fitness_func_values = np.zeros(self.population_size)
        self.fitness_func_values_copy = np.zeros(self.population_size)
        self.fitness_func_values_front = np.zeros(self.n_inds_front)
        # Параметры алгоритма
        self.weights = np.zeros(self.population_size)
        self.temp_success_Cr = np.zeros(self.population_size)
        self.fitness_values_dif = np.zeros(self.population_size)
        self.Cr_memory = np.ones(self.memory_size)
        self.trial_solution = np.zeros(self.n_vars)
        self.indices = np.zeros(self.population_size, dtype=int)
        self.indices2 = np.zeros(self.population_size, dtype=int)

        # Шаг 5: Генерация начальной популяции
        self.population = np.random.uniform(
            self.left,
            self.right,
            size=(self.population_size, self.n_vars)
        )

    def update_Cr_memory(self):
        """
        Функция отвечает за адаптацию параметра кроссовера (Cr)
        на основе успешных значений, найденных в текущем поколении.
        """
        if self.success_filled != 0:
            self.Cr_memory[self.memory_iter] += 0.5 * self.mean_wl(
                self.temp_success_Cr, self.fitness_values_dif
            )
            self.memory_iter = (self.memory_iter + 1) % self.memory_size

    def mean_wl(
            self,
            array: np.ndarray | List,
            temp_weights: np.ndarray | List
    ) -> float:
        """
        Функция возвращает адаптивное значение Cr,
        основанное на успешных значениях Cr
        из предыдущего поколения, с учетом того,
        насколько они улучшили результат.
        """
        sum_weight = 0
        sum_square = 0
        _sum = 0
        weights = np.empty(self.success_filled)

        for i in range(self.success_filled):
            sum_weight += temp_weights[i]

        for i in range(self.success_filled):
            if sum_weight == 0 or np.isnan(sum_weight):
                weights[i] = 0
            else:
                weights[i] = temp_weights[i] / sum_weight

        for i in range(self.success_filled):
            sum_square += weights[i] * np.power(array[i], 2)
            _sum += weights[i] * array[i]

        if abs(_sum) > 1e-8:
            return sum_square / _sum
        else:
            return 1.0

    def find_n_save_best(self, init, ind_iter):
        """
        Функция обновляет локальные
        и глобальные лучшие значения целевой функции.
        """
        if self.fitness_func_values[ind_iter] <= self.best_fitness_value or init:
            self.best_fitness_value = self.fitness_func_values[ind_iter]
        if self.best_fitness_value < self._global_variables.global_best or init:
            self._global_variables.global_best = self.best_fitness_value

    def save_best_values(self):
        """
        Функция предназначена для сохранения сходимости
        траектории алгоритма оптимизации.
        """
        temp = self._global_variables.global_best - self._global_variables.eval_func_opt_value
        if temp <= 1e-8 \
                and self._global_variables.results_array[
            self._consts.records_number_per_function - 1
        ] == self._global_variables.max_eval_func_calls:
            self._global_variables.results_array[
                self._consts.records_number_per_function - 1
                ] = self._global_variables.eval_func_calls

        for step_eval_func_count in range(
                self._global_variables.last_eval_step,
                self._consts.records_number_per_function - 1):
            if self._global_variables.eval_func_calls == self._global_variables.eval_steps[
                step_eval_func_count
            ]:
                if temp <= 1e-8:
                    temp = 0
                self._global_variables.results_array[step_eval_func_count] = temp
                self._global_variables.last_eval_step = step_eval_func_count
                self._global_variables.best_of_best_array[step_eval_func_count] = self._global_variables.global_best

    @staticmethod
    def reflect(val, left, right):
        """Отражение элемента при лимитах границ."""
        if val < left or val > right:
            span = right - left
            val = left + abs((val - left) % (2 * span))
            if val > right:
                val = right - (val - right)
        return val

    def remove_worst(self, _n_inds_front: int, new_n_inds_front: int):
        """
        Удаление худших особей из популяции
        по значению целевой функции.
        Шаг 8.I
        """
        points_to_remove = _n_inds_front - new_n_inds_front

        for l in range(points_to_remove):
            worst = np.argmax(self.fitness_func_values_front[:_n_inds_front])

            # сдвиг
            self.front_population[
                worst:_n_inds_front - 1
            ] = self.front_population[worst + 1:_n_inds_front]
            self.fitness_func_values_front[
                worst:_n_inds_front - 1
            ] = self.fitness_func_values_front[worst + 1:_n_inds_front]
            _n_inds_front -= 1

            # «очистить» хвост
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

    def main_cycle(self):
        """Запуск алгоритма L-SRTDE."""
        # Шаг 6: Инициализация fitness значений для стартовой популяции
        for i_inds in range(self.n_inds_front):
            self.fitness_func_values[i_inds] = self.fitness_function(self.population[i_inds])
            self._global_variables.eval_func_calls += 1
            self.find_n_save_best(i_inds == 0, i_inds)

            if (not self._global_variables.global_best_init
                    or self.best_fitness_value < self._global_variables.global_best):
                self._global_variables.global_best = self.best_fitness_value
                self._global_variables.global_best_init = True

            self.save_best_values()

        # Шаг 7: Копирование значений целевых функций и индексов
        # Определение мин/макс целевой функции
        for i in range(self.n_inds_front):
            self.fitness_func_values_copy[i] = self.fitness_func_values[i].copy()
            self.indices[i] = i

        min_fitness_value = np.min(self.fitness_func_values[:self.n_inds_front])
        max_fitness_value = np.max(self.fitness_func_values[:self.n_inds_front])

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

        # Шаг 8: Главный цикл алгоритма
        # Пока не дойдем до максимума вычислений целевой функции
        while self._global_variables.eval_func_calls < self._global_variables.max_eval_func_calls:

            _max_calls = self._global_variables.max_eval_func_calls
            _current_calls = self._global_variables.eval_func_calls

            if self.verbose:
                print(
                    f'--- Iteration {self.iter_number}, '
                    f'left {_max_calls - _current_calls} calls, '
                    f'current best fitness value: {self._global_variables.global_best}'
                )

            # Расчет параметров
            # Шаг 8.D.a
            mean_F = 0.4 + np.tanh(self.success_rate * 5) * 0.25
            sigma_F = 0.02

            # Копируем и сортируем значения целевой функции для всей популяции
            for i in range(self.n_inds_front):
                self.fitness_func_values_copy[i] = self.fitness_func_values[i]
                self.indices[i] = i

            min_fitness_value = np.min(self.fitness_func_values[:self.n_inds_front])
            max_fitness_value = np.max(self.fitness_func_values[:self.n_inds_front])

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

            min_fitness_value = np.min(self.fitness_func_values_front[:self.n_inds_front])
            max_fitness_value = np.max(self.fitness_func_values_front[:self.n_inds_front])

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

            # Шаг 8.D.h
            # Частично 8.D.i.i
            p_size_val = max(
                2,
                int(self.n_inds_front * 0.7 * np.exp(-self.success_rate * 7))
            )

            # Шаг 8: Итерируемся по фронтовой части популяции
            for i_ind in range(self.n_inds_front):

                # Выбираем индекс из фронтовой части популяции
                self.chosen_index = self._random_generators.random_integers(
                    low=0, high=self.n_inds_front - 1
                )
                # Выбираем индекс выбранного индивида в координатах population
                chosen_pop_idx = self.indices2[self.chosen_index]

                # Выбираем индекс из памяти
                # Шаг 8.D.d
                self.memory_index = self._random_generators.random_integers(
                    low=0, high=self.memory_size - 1
                )

                # Выбор p_rand, не равного индексу выбранного индивида
                # Шаг 8.D.i.i
                p_rand = self.indices[
                    self._random_generators.random_integers(
                        low=0, high=p_size_val - 1
                    )
                ]
                while p_rand == chosen_pop_idx:
                    p_rand = self.indices[
                        self._random_generators.random_integers(
                            low=0, high=p_size_val - 1
                        )
                    ]

                # Выбор Rand1, не равного p_rand
                # Шаг 8.D.g
                Rand1 = self.indices2[
                    self._random_generators.random_integers(
                        low=0, high=self.n_inds_front - 1
                    )
                ]
                Rand1_pop = self.indices2[Rand1]
                while Rand1_pop == p_rand or Rand1_pop == chosen_pop_idx:
                    Rand1 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=self.n_inds_front - 1
                        )
                    ]
                    Rand1_pop = self.indices2[Rand1]

                # Выбор Rand2, не равного p_rand и Rand1
                # Шаг 8.D.i.ii
                Rand2 = self.indices2[component_selector_front()]
                Rand2_pop = self.indices2[Rand2]
                while Rand2_pop == p_rand or Rand2_pop == Rand1_pop or Rand2_pop == chosen_pop_idx:
                    Rand2 = self.indices2[component_selector_front()]
                    Rand2_pop = self.indices2[Rand2]

                # Выбор Rand2, не равного p_rand, Rand1 и Rand2
                # Шаг 8.D.i.iii
                Rand3 = self.indices2[
                    self._random_generators.random_integers(
                        low=0, high=self.n_inds_front - 1
                    )
                ]
                while Rand3 == p_rand or Rand3 == Rand1_pop or Rand3 == Rand2_pop or Rand3 == chosen_pop_idx:
                    Rand3 = self.indices2[
                        self._random_generators.random_integers(
                            low=0, high=self.n_inds_front - 1
                        )
                    ]

                # Генерация F и Cr с ограничениями
                # Параметр мутации F
                # Шаг 8.D.b-8.D.c
                self.F = np.random.normal(mean_F, sigma_F)
                while self.F < 0.0 or self.F > 1.0:
                    self.F = np.random.normal(mean_F, sigma_F)

                # Параметр кроссовера Cr из памяти
                # Шаг 8.D.e
                self.Cr = np.random.normal(self.Cr_memory[self.memory_index], 0.05)
                # Ограничиваем Cr диапазоном [0, 1]
                # Шаг 8.d.f
                self.Cr = min(max(self.Cr, 0.0), 1.0)
                actual_Cr = 0
                Will_crossover = random.randint(0, self.n_vars - 1)

                # Создаем массив для пробного решения
                self.trial_solution = np.zeros(self.n_vars)

                for j in range(self.n_vars):
                    if self._random_generators.random_floats() < self.Cr or Will_crossover == j:
                        val = (self.population[Rand1].reshape(-1)[j]
                               + self.F * (self.front_population[p_rand].reshape(-1)[j]
                                           - self.population[self.chosen_index].reshape(-1)[j])
                               + self.F * (self.population[Rand2].reshape(-1)[j]
                                           - self.front_population[Rand3].reshape(-1)[j]))

                        val = self.reflect(val, self.left, self.right)
                        self.trial_solution[j] = val
                        actual_Cr += 1
                    else:
                        self.trial_solution[j] = self.front_population[self.chosen_index].reshape(-1)[j]

                actual_Cr /= float(self.n_vars)

                # Считаем значение целевой функции для пробного решения
                temp_fit = self.fitness_function(self.trial_solution)
                self._global_variables.eval_func_calls += 1

                if temp_fit <= self.fitness_func_values_front[self.chosen_index]:
                    # Если улучшение найдено
                    idx = self.n_inds_front + self.success_filled
                    self.population[idx] = self.trial_solution.copy()
                    self.front_population[self.pf_index] = self.trial_solution.copy()
                    self.fitness_func_values[idx] = temp_fit
                    self.fitness_func_values_front[self.pf_index] = temp_fit
                    # Обновляем лучшее решение
                    self.find_n_save_best(False, idx)
                    # Сохраняем Cr
                    self.temp_success_Cr[self.success_filled] = actual_Cr
                    self.fitness_values_dif[self.success_filled] = abs(
                        self.fitness_func_values_front[self.chosen_index] - temp_fit
                    )

                    self.success_filled += 1
                    # Сдвигаем индекс pf_index
                    self.pf_index = (self.pf_index + 1) % self.n_inds_front

                self.save_best_values()

            # Вычисляем процент успешных мутаций
            self.success_rate = float(self.success_filled) / float(self.n_inds_front)

            # Вычисляем новый размер фронтовой части популяции
            self.new_n_inds_front = int(
                (4 - self.n_inds_front_max) / self._global_variables.max_eval_func_calls
                * self._global_variables.eval_func_calls
                + self.n_inds_front_max
            )
            # Удаляем худшие решения из фронтовой части популяции
            self.remove_worst(self.n_inds_front, self.new_n_inds_front)
            # Обновляем размер фронтовой части популяции
            self.n_inds_front = self.new_n_inds_front
            # Обновляем память Cr
            self.update_Cr_memory()
            # Обновляем текущий размер популяции
            self.n_inds_current = self.n_inds_front + self.success_filled
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

        # Аналитика результатов после завершения оптимизации

        print(f'--- Global best: {self._global_variables.global_best}')

        return None, self._global_variables.global_best
