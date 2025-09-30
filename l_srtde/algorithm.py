import os
import time
import warnings
from pathlib import Path
from typing import List, Callable, Union

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, MT19937


class AlgorithmConst:
    """
    Константы для алгоритма.

    Аттрибуты:
        records_number_per_function: количество точек данных, сохраненных в массиве результатов.
        global_seed: глобальный seed для всех генераторов.
    """
    records_number_per_function = 1001
    global_seed = 2025


class AlgorithmGlobals:
    """
    Глобальные переменные для алгоритма.

    Аттрибуты:
        eval_steps: массив контрольных значений оценки, в котором будет храниться ошибка оптимизации.
        result_array: хранит значения оптимума.
        sr_array: хранит значения истории успешности.
        last_eval_step: счетчик для хранения последнего шага оценки.
        eval_func_calls: счетчик для хранения количества текущих вызовов целевой функции.
        max_eval_func_calls: максимальное количество вызовов целевой функции (ограничение).
        problem_dimension: размерность задачи.
        eval_func_opt_value: оптимальное значение целевой функции.
        global_best: наилучшее значение целевой функции, найденное глобально (в текущей итерации).
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



class AlgorithmRandomGenerators:
    """Класс, реализующий генераторы случайных чисел."""

    def __init__(self, globalseed: int):
        seeds = [globalseed, globalseed + 100, globalseed + 200, globalseed + 300]

        # Четыре независимых генератора
        self.generator_uni_i = Generator(MT19937(seeds[0]))
        self.generator_uni_r = Generator(MT19937(seeds[1]))
        self.generator_norm = Generator(MT19937(seeds[2]))
        self.generator_uni_i_2 = Generator(MT19937(seeds[3]))

    def int_random(self, target: int) -> Union[np.signedinteger,int]:
        """Целое число."""
        if target == 0:
            return 0
        return self.generator_uni_i.integers(low=0, high=target)

    def random(self, minimal: float, maximal: float) -> float:
        """Число с плавающей точкой."""
        rand01 = self.generator_uni_r.random()
        return rand01 * (maximal - minimal) + minimal

    def norm_rand(self, mu: float, sigma: float) -> float:
        """Нормальное распределение, плавающая точка."""
        val = self.generator_norm.normal(loc=0.0, scale=1.0)
        return val * sigma + mu

    def int_random2(self, target: int) -> Union[np.signedinteger,int]:
        """Аналог использования int_random."""
        if target == 0:
            return 0
        return self.generator_uni_i_2.integers(low=0, high=target)

    def create_weighted_selector(self, weights: Union[List, np.ndarray]):
        """
        Функция создает функцию-генератор, которая возвращает один индекс, выбранный случайным образом с учетом весов.
        """
        weights_arr = np.array(weights, dtype=float)
        probs = weights_arr / weights_arr.sum()
        rng = self.generator_uni_i_2
        def selector():
            return rng.choice(len(weights_arr), p=probs)
        return selector

    def choices(self, elements: List, k: int = 1):
        """Аналог random.choices(elements, k=k)."""
        rng = self.generator_uni_r
        elements_arr = np.array(elements, dtype=object)
        return rng.choice(elements_arr, size=k, replace=True).tolist()


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
    globals = AlgorithmGlobals()
    _generators = AlgorithmRandomGenerators(consts.global_seed)

    threshold = 1e-8

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
        warnings.filterwarnings("ignore")

        self.run_id = int(round(time.time(), 5)*100_000)
        self.tmp_folder = Path('tmp', f'tmp_{self.run_id}')

        self.is_last_call = False
        self.t0 = time.time()
        self.verbose = verbose

        if self.verbose:
            os.makedirs('tmp', exist_ok=True)
            os.makedirs(self.tmp_folder, exist_ok=True)

            # Массивы и файлы для хранения всех изменений в параметрах
            self.tmp_eval_calls = []
            open(Path(self.tmp_folder, 'eval_calls'), 'a').close()
            self.tmp_f = []
            open(Path(self.tmp_folder, 'f'), 'a').close()
            self.tmp_cr = []
            open(Path(self.tmp_folder, 'cr'), 'a').close()
            self.tmp_front = []
            open(Path(self.tmp_folder, 'front'), 'a').close()
            self.tmp_sr = []
            open(Path(self.tmp_folder, 'sr'), 'a').close()
            self.tmp_sf = []
            open(Path(self.tmp_folder, 'sf'), 'a').close()

        # Инициализация целевой функции
        self.fitness_function = fitness_function

        # Счетчики и локальные переменные
        self.memory_size = 5
        self.memory_iter = 0
        self.memory_index = 0
        self.success_filled = 0

        # Обнуление глобальных счетчиков
        self.globals.eval_func_calls = 0
        self.globals.last_eval_step = 0
        self.globals.global_best = np.inf

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
        self.mean_F = 0.0
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
        self.globals.eval_steps = [
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
            new_cr = 0.5 * (
                    self.mean_wl(
                        self.temp_success_cr, self.fitness_values_dif
                    ) + self.cr_memory[self.memory_iter]
            )
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

        _limit = self.success_filled
        sum_weight = np.sum(temp_weights[:_limit])
        self.weights = temp_weights[:_limit] / sum_weight
        sum_square = np.sum(self.weights[:_limit] * array[:_limit] * array[:_limit])
        sum_val = np.sum(self.weights[:_limit] * array[:_limit])

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
        if self.best_fitness_value < self.globals.global_best or init:
            self.globals.global_best = self.best_fitness_value

    def save_best_values(self):
        """
        Функция предназначена для сохранения сходимости
        траектории алгоритма оптимизации.
        """
        temp = self.globals.global_best - self.globals.eval_func_opt_value
        if temp <= self.threshold and self.globals.result_array[
            self.consts.records_number_per_function - 1
        ] == self.globals.max_eval_func_calls:
            self.globals.result_array[
                self.consts.records_number_per_function - 1
                ] = self.globals.eval_func_calls

        for step_eval_func_count in range(
                self.globals.last_eval_step,
                self.consts.records_number_per_function - 1
        ):
            if self.globals.eval_func_calls == self.globals.eval_steps[
                step_eval_func_count
            ]:
                if temp <= self.threshold:
                    temp = 0

                # Обновление результирующих массивов
                self.globals.result_array[step_eval_func_count] = temp
                self.globals.sr_array[step_eval_func_count] = self.success_rate

                # Сохраняем текущие значения параметров
                self.tmp_eval_calls.append(self.globals.eval_func_calls)
                self.tmp_f.append(round(self.mean_F, 2))
                self.tmp_cr.append(round(self.cr, 2))
                self.tmp_front.append(self.n_inds_front)
                self.tmp_sr.append(round(self.success_rate, 3))
                self.tmp_sf.append(self.success_filled)

                self.globals.last_eval_step = step_eval_func_count

    def remove_worst(self, _n_inds_front: int, new_n_inds_front: int):
        """
        Удаление худших особей из популяции
        по значению целевой функции.
        """
        points_to_remove = _n_inds_front - new_n_inds_front

        for l in range(points_to_remove):
            worst = np.argmax(self.fitness_func_values_front[:_n_inds_front])

            # Сдвиг
            self.front_population[worst:_n_inds_front-1] = self.front_population[worst+1:_n_inds_front]
            self.fitness_func_values_front[worst:_n_inds_front-1] \
                = self.fitness_func_values_front[worst+1:_n_inds_front]
            _n_inds_front -= 1

            # «Очистить» хвост
            self.front_population[new_n_inds_front:] = 0.0
            self.fitness_func_values_front[new_n_inds_front:] = np.inf
        return _n_inds_front

    @staticmethod
    def minmax(array, limit):
        min_value = np.min(array[:limit])
        max_value = np.max(array[:limit])
        return min_value, max_value

    def __name__(self):
        return self.run_id

    def __call__(self):
        """Запуск алгоритма L-SRTDE."""
        # Инициализация значений целевой функции для стартовой популяции
        for i_inds in range(self.n_inds_front):
            self.fitness_func_values[i_inds] = self.fitness_function(self.population[i_inds])
            self.globals.eval_func_calls += 1
            self.find_n_save_best(i_inds == 0, i_inds)

            if (not self.globals.global_best == np.inf
                    or self.best_fitness_value < self.globals.global_best):
                self.globals.global_best = self.best_fitness_value
            self.save_best_values()

        # Копирование значений целевых функций и индексов
        for i in range(self.n_inds_front):
            self.fitness_func_values_copy[i] = self.fitness_func_values[i].copy()
            self.indices[i] = i

        # Определение мин/макс целевой функции        
        min_fit_val, max_fit_val = self.minmax(array=self.fitness_func_values, limit=self.n_inds_front)

        # Сортировка и обновление фронтовой части популяции
        if min_fit_val != max_fit_val:
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
        while self.globals.eval_func_calls < self.globals.max_eval_func_calls:
            epoch_num += 1

            _dif_calls = self.globals.max_eval_func_calls - self.globals.eval_func_calls

            _max_calls = self.globals.max_eval_func_calls
            _current_calls = self.globals.eval_func_calls

            if self.verbose and self.iter_number % (self.globals.max_eval_func_calls // 1_000) == 0:
                self.t0 = time.time()

            # Расчет параметров
            self.mean_F = 0.4 + np.tanh(self.success_rate * 5) * 0.25
            sigma_F = 0.02  

            # Копируем и сортируем значения целевой функции для всей популяции
            for i in range(self.n_inds_front):
                self.fitness_func_values_copy[i] = self.fitness_func_values[i]
                self.indices[i] = i

            min_fit_val, max_fit_val = self.minmax(array=self.fitness_func_values, limit=self.n_inds_front)

            if min_fit_val != max_fit_val:
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

            min_fit_val, max_fit_val = self.minmax(
                array=self.fitness_func_values_front,
                limit=self.n_inds_front
            )

            if min_fit_val != max_fit_val:
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
            component_selector_front = self._generators.create_weighted_selector(fitness_temp2)

            p_size_val = max(
                2,
                int(self.n_inds_front * 0.7 * np.exp(-self.success_rate * 7))
            )

            # Итерируемся по фронтовой части популяции
            for i_ind in range(self.n_inds_front):
                # Выбираем индекс из фронтовой части популяции
                self.chosen_index = self._generators.int_random(self.n_inds_front-1)

                # Выбираем индекс выбранного индивида в координатах population
                chosen_pop_idx = self.indices2[self.chosen_index]

                # Выбираем индекс из памяти
                self.memory_index = self._generators.int_random(self.memory_size-1)

                # Выбор p_rand, не равного индексу выбранного индивида
                p_rand = self.indices[self._generators.int_random(p_size_val-1)]
                max_at, at = 100, 0
                while p_rand == chosen_pop_idx and max_at < at:
                    p_rand = self.indices[self._generators.int_random(p_size_val-1)]
                    at += 1
                if at == max_at:
                    p_rand = self.indices[self._generators.int_random(p_size_val-1)]

                # Выбор rand1, не равного p_rand
                rand1 = self.indices2[self._generators.int_random(self.n_inds_front-1)]
                rand1_pop = self.indices2[rand1]
                max_at, at = 100, 0
                while (rand1_pop == p_rand or rand1_pop == chosen_pop_idx) and max_at < at:
                    rand1 = self.indices2[self._generators.int_random(self.n_inds_front-1)]
                    rand1_pop = self.indices2[rand1]
                    at += 1
                if at == max_at:
                    rand1 = self.indices2[self._generators.int_random(p_size_val-1)]

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
                rand3 = self.indices2[self._generators.int_random(self.n_inds_front-1)]
                max_at, at = 100, 0
                while (rand3 == p_rand or rand3 == rand1_pop
                       or rand3 == rand2_pop or rand3 == chosen_pop_idx) and max_at < at:
                    rand3 = self.indices2[self._generators.int_random(self.n_inds_front-1)]
                    at += 1
                if at == max_at:
                    rand3 = self.indices2[self._generators.int_random(self.n_inds_front-1)]

                # Генерация F и cr с ограничениями
                # Параметр мутации F
                self.f = np.clip(self._generators.norm_rand(self.mean_F, sigma_F), 0.0, 1.0)

                # Параметр кроссовера cr из памяти
                self.cr = self._generators.norm_rand(float(self.cr_memory[self.memory_index]), 0.05)

                # Ограничиваем cr диапазоном [0, 1]
                self.cr = min(max(self.cr, 0.0), 1.0)
                actual_cr = 0
                will_crossover = self._generators.int_random(self.n_vars-1)

                # Создаем массив для пробного решения
                self.trial_solution = np.zeros(self.n_vars)

                for j in range(self.n_vars):
                    if self._generators.random(0., 1.) < self.cr or will_crossover == j:
                        val = (self.population[rand1].reshape(-1)[j]
                               + self.f * (self.front_population[p_rand].reshape(-1)[j]
                                           - self.population[self.chosen_index].reshape(-1)[j])
                               + self.f * (self.population[rand2].reshape(-1)[j]
                                           - self.front_population[rand3].reshape(-1)[j]))

                        if val < self.left:
                            val = self._generators.choices([self.left, self.right])[0]
                        if val > self.right:
                            val = self._generators.choices([self.left, self.right])[0]

                        self.trial_solution[j] = val
                        actual_cr += 1
                    else:
                        self.trial_solution[j] = self.front_population[self.chosen_index].reshape(-1)[j]

                actual_cr /= float(self.n_vars)

                # Считаем значение целевой функции для пробного решения
                temp_fit = self.fitness_function(self.trial_solution)
                self.globals.eval_func_calls += 1

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

            # Вычисляем процент успешных мутаций
            self.success_rate = float(self.success_filled) / float(self.n_inds_front)

            # Вычисляем новый размер фронтовой части популяции
            self.new_n_inds_front = int(
                (4 - self.n_inds_front_max) / self.globals.max_eval_func_calls
                * self.globals.eval_func_calls
                + self.n_inds_front_max
            )

            # Удаляем худшие решения из фронтовой части популяции
            self.remove_worst(self.n_inds_front, self.new_n_inds_front)
            # Обновляем размер фронтовой части популяции
            self.n_inds_front = self.new_n_inds_front
            # Обновляем память cr
            self.update_cr_memory()
            # Обновляем текущий размер популяции
            self.n_inds_current = self.n_inds_front + self.success_filled

            if self.verbose:
                # Выводим на экран текущие значения параметров
                if epoch_num % 1000 == 0 and epoch_num > 0:
                    print(f'\t {self.run_id} | Epoch {epoch_num} | eval calls: {self.globals.eval_func_calls} '
                          f'| Current Optimum: {round(self.globals.global_best,3)} '
                          f'| F: {round(self.mean_F,2)} | Cr: {round(self.cr,2)} | Front size: {self.n_inds_front} '
                          f'| SR: {round(self.success_rate,3)} | SF: {self.success_filled}')
            # Обнуляем количество успешных мутаций
            self.success_filled = 0
            # Увеличиваем счетчик поколений
            self.iter_number += 1

            # Коррекция популяции, если она увеличилась
            if self.n_inds_current > self.n_inds_front:
                for i in range(self.n_inds_current):
                    self.indices[i] = i

                min_fit_val, max_fit_val = self.minmax(
                    array=self.fitness_func_values,
                    limit=self.n_inds_current
                )

                if min_fit_val != max_fit_val:
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

        if self.verbose:
            with open(Path(self.tmp_folder, 'eval_calls'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_eval_calls]))
            with open(Path(self.tmp_folder, 'f'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_f]))
            with open(Path(self.tmp_folder, 'cr'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_cr]))
            with open(Path(self.tmp_folder, 'front'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_front]))
            with open(Path(self.tmp_folder, 'sr'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_sr]))
            with open(Path(self.tmp_folder, 'sf'), 'r+') as f:
                f.write(' '.join([str(x) for x in self.tmp_sf]))

        return self.run_id, self.globals.global_best
