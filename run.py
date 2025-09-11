import os
import time
import argparse
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from l_srtde.algorithm import Algorithm
from _gnbg_functions.gnbg import GNBG


if __name__ == '__main__':
    # Парсинг аргументов для запуска
    parser = argparse.ArgumentParser(prog='populationes-runs')
    parser.add_argument('--runNum', type=int, default=30)
    parser.add_argument('--funcNum', type=int, default=24)
    parser.add_argument('--populationSize', type=int, default=10)
    parser.add_argument('--displayEpochLimit', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(Path('l_srtde', 'Results_Python_implementation'), exist_ok=True)
    # Основные настройки

    # Количество прогонов для каждой функции
    total_n_runs = args.runNum
    # Количество функций
    max_n_funcs = args.funcNum
    # Размер популяции
    population_size = args.populationSize

    is_need_sr_save = False

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
                print(f"Running algorithm on {gnbg}, run {run + 1}")

                # Инициализация алгоритма
                optz = Algorithm(
                    fitness_function=gnbg.fitness,
                    population_size=population_size * gnbg.dimension,
                    problem_dimension=gnbg.dimension,
                    verbose=False
                )

                # Настраиваем максимальное количество вызовов целевой функции
                optz.global_variables.max_eval_func_calls = gnbg.max_evals
                optz.global_variables.result_array[
                    optz.consts.records_number_per_function-1
                ] = gnbg.max_evals
                _, optimum = optz()

                # Читаем результаты работы
                _optimums[run] = optimum
                res[func_num, run] = optz.global_variables.result_array
                sr_res[func_num, run] = optz.global_variables.sr_array

                print(f'Found optimum: {optimum}')
            optimums[func_num] = np.mean(_optimums)

            # Сохраняем результаты работы
            np.savetxt(Path(
                'l_srtde', 'Results_Python_implementation',
                f'L-SRTDE_GNBG_F{func_num+1}_D{gnbg.dimension}.txt'
            ), res[func_num], fmt='%.1f')
            if is_need_sr_save:
                np.savetxt(Path(
                    'l_srtde', 'Results_Python_implementation', f'F{func_num+1}_Success_Rate.txt'
                ), sr_res[func_num], fmt='%.1f')

        except Exception as e:
            print(f'\t### Error in function {func_num}: {e}')
            print(f'\t### Traceback: {traceback.format_exc()}')

    print(f'Elapsed time: {round(time.time() - t0, 2)} sec')

    is_need_log = False
    is_need_saving = True

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    gs = fig.add_gridspec(6, 4)
    # Количество отображаемых эпох
    limit = args.displayEpochLimit

    # Визуализация
    for func_num in range(max_n_funcs):
        row = func_num // 4
        col = func_num % 4
        ax = fig.add_subplot(gs[row, col])
        optimum = float(optimums[func_num])
        for run in range(total_n_runs):
            if is_need_log:
                ax.semilogy(res[func_num, run, :limit], label=f"Run {run + 1}: {round(optimum, 2)}")
            else:
                ax.plot(res[func_num, run, :limit], label=f"Run {run + 1}: {round(optimum, 2)}")

        ax.set_title(f"F{func_num + 1}", fontsize=16)
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Best Fitness")

        ax.axhline(
            y=round(float(fopts[func_num]), 3),
            color='red',
            linestyle='--',
            label=f'Real optimum: {round(float(fopts[func_num]), 3)}'
        )

        ax.grid(True)
        ax.legend(fontsize='medium')

    plt.tight_layout()
    if is_need_saving:
        plt.savefig(Path('l_srtde', 'python_l_srtde_on_gnbg.png'))
    plt.show()
