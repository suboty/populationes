import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from l_srtde.algorithm import Algorithm
from _gnbg_functions.gnbg_for_json import GNBG

if __name__ == '__main__':
    # Основные настройки

    # Количество прогонов для каждой функции
    total_n_runs = 1
    # Количество функций
    max_n_funcs = 1
    # Размер популяции
    population_size = 10

    # Ошибки
    res_errors = np.zeros((max_n_funcs, total_n_runs, 1001))
    # Минимумы
    res = np.zeros((max_n_funcs, total_n_runs, 1001))
    # Показатели успешности (Success Rate)
    sr_res = np.zeros((max_n_funcs, total_n_runs, 1001))

    # Найденные оптимумы
    optimums = np.zeros(max_n_funcs)

    # Необходимые оптимумы
    fopts = np.zeros(max_n_funcs)

    # Запуск алгоритма и сбор результатов
    t0 = time.time()
    for func_num in range(0, max_n_funcs):
        print(f'Func {func_num + 1}')
        try:
            gnbg = GNBG(func_num + 1)
            fopt = gnbg.optimum_value
            print(f'Real optimum: {fopt}')
            fopts[func_num] = fopt

            _optimums = np.zeros(total_n_runs)

            for run in range(total_n_runs):
                print(f"\tRunning algorithm on {gnbg}, run {run + 1}")

                # Инициализация алгоритма
                optz = Algorithm(
                    fitness_function=gnbg.fitness,
                    population_size=population_size * gnbg.dimension,
                    problem_dimension=gnbg.dimension,
                    verbose=False
                )

                # Настраиваем максимальное количество вызовов целевой функции
                optz._global_variables.max_eval_func_calls = gnbg.max_evals
                _, optimum = optz()

                # Читаем результаты работы
                _optimums[run] = optimum
                res_errors[func_num, run] = optz._global_variables.error_array
                res[func_num, run] = optz._global_variables.best_of_best_array
                sr_res[func_num, run] = optz._global_variables.sr_array

                print(f'\tFinded optimum: {optimum}')
            optimums[func_num] = np.mean(_optimums)
        except Exception as e:
            print(f'\t### Error in function {func_num}: {e}')
            print(f'\t### Traceback: {traceback.format_exc()}')

    print(f'Elapsed time: {round(time.time() - t0, 2)} sec')

    is_need_log = False
    is_need_saving = False

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    # Количество отображаемых эпох
    limit = 500

    # Визуализация
    for func_num in range(max_n_funcs):
        row = func_num // 4
        col = func_num % 4
        ax = fig.add_subplot(gs[row, col])
        optimum = optimums[func_num]
        for run in range(total_n_runs):
            if is_need_log:
                ax.semilogy(res[func_num, run, :limit], label=f"Run {func_num + 1} gets optimum: {optimum}")
            else:
                ax.plot(res[func_num, run, :limit], label=f"Run {func_num + 1} gets optimum: {optimum}")

        ax.set_title(f"F{func_num + 1}", fontsize=16)
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Best Fitness")

        ax.axhline(
            y=fopts[func_num],
            color='red',
            linestyle='--',
            label=f'Real optimum: {round(fopts[func_num], 2)}'
        )

        ax.grid(True)
        ax.legend(fontsize='large')

    if is_need_saving:
        plt.savefig('python_l_strde_on_gnbg.png')
    plt.show()
