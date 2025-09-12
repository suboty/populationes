import os
import time
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from l_srtde.algorithm import Algorithm
from _gnbg_functions.gnbg import GNBG


def run_algorithm_for_func(func_num, total_n_runs, population_size):
    result_dict = {}
    try:
        gnbg = GNBG(func_num + 1)
        fopt = gnbg.optimum_value
        print(f'Func {func_num + 1} | Real optimum: {fopt}')

        _optimums = np.zeros(total_n_runs)
        res = np.zeros((total_n_runs, 1001))
        sr_res = np.zeros((total_n_runs, 1001))

        for run in range(total_n_runs):
            print(f"Func {func_num + 1} | Running algorithm, run {run + 1}")
            optz = Algorithm(
                fitness_function=gnbg.fitness,
                population_size=population_size * gnbg.dimension,
                problem_dimension=gnbg.dimension,
                verbose=False
            )
            optz.global_variables.eval_func_opt_value = fopt
            optz.global_variables.max_eval_func_calls = gnbg.max_evals
            optz.global_variables.result_array[
                optz.consts.records_number_per_function-1
            ] = gnbg.max_evals
            _, optimum = optz()

            _optimums[run] = optimum
            res[run] = optz.global_variables.result_array
            sr_res[run] = optz.global_variables.sr_array
            print(f'Func {func_num + 1} | Run {run + 1} finished, Found optimum: {optimum}')

        result_dict = {
            'func_num': func_num,
            'res': res,
            'sr_res': sr_res,
            'optimums': _optimums,
            'mean_optimum': np.mean(_optimums),
            'fopt': fopt,
            'dimension': gnbg.dimension
        }

    except Exception as e:
        print(f'### Error in function {func_num}: {e}')
        print(traceback.format_exc())
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='populationes-runs')
    parser.add_argument('--runNum', type=int, default=31)
    parser.add_argument('--funcNum', type=int, default=24)
    parser.add_argument('--populationSize', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(Path('l_srtde', 'Results_Python_implementation'), exist_ok=True)

    total_n_runs = args.runNum
    max_n_funcs = args.funcNum
    population_size = args.populationSize

    res = np.zeros((max_n_funcs, total_n_runs, 1001))
    sr_res = np.zeros((max_n_funcs, total_n_runs, 1001))
    optimums = np.zeros(max_n_funcs)
    fopts = np.zeros(max_n_funcs)

    t0 = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_algorithm_for_func,
                func_num, total_n_runs, population_size
            ) for func_num in range(max_n_funcs)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                func_num = result['func_num']
                res[func_num] = result['res']
                sr_res[func_num] = result['sr_res']
                optimums[func_num] = result['mean_optimum']
                fopts[func_num] = result['fopt']

                np.savetxt(Path(
                    'l_srtde', 'Results_Python_implementation',
                    f'L-SRTDE_GNBG_F{func_num+1}_D{result["dimension"]}.txt'
                ), res[func_num], fmt='%.1f')

    print(f'Elapsed time: {round(time.time() - t0, 2)} sec')
