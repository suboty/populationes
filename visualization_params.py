import copy
import json
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_params_template = {
    'eval_calls': [],
    'f': [],
    'cr': [],
    'front': [],
    'sr': [],
    'sf': [],
}

window = 10


if __name__ == '__main__':
    with open(Path('tmp', 'run_ids.json'), 'r') as f:
        run_ids = json.load(f)

    for func_num in run_ids.keys():
        runs = copy.deepcopy(_params_template)
        for run_num in run_ids[func_num]:
            with open(Path(f'tmp', f'tmp_{run_num}', 'eval_calls'), 'r') as f:
                runs['eval_calls'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
            with open(Path(f'tmp', f'tmp_{run_num}', 'f'), 'r') as f:
                runs['f'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
            with open(Path(f'tmp', f'tmp_{run_num}', 'cr'), 'r') as f:
                runs['cr'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
            with open(Path(f'tmp', f'tmp_{run_num}', 'front'), 'r') as f:
                runs['front'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
            with open(Path(f'tmp', f'tmp_{run_num}', 'sr'), 'r') as f:
                runs['sr'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
            with open(Path(f'tmp', f'tmp_{run_num}', 'sf'), 'r') as f:
                runs['sf'].append(np.loadtxt(
                    StringIO(f.readline().rstrip()),
                ))
        runs = {
            'eval_calls': np.convolve(np.mean(runs['eval_calls'], axis=0), np.ones(window)/window, 'valid'),
            'f': np.convolve(np.mean(runs['f'], axis=0), np.ones(window)/window, 'valid'),
            'cr': np.convolve(np.mean(runs['cr'], axis=0), np.ones(window)/window, 'valid'),
            'front': np.convolve(np.mean(runs['front'], axis=0), np.ones(window)/window, 'valid'),
            'sr': np.convolve(np.mean(runs['sr'], axis=0), np.ones(window)/window, 'valid'),
            'sf': np.convolve(np.mean(runs['sf'], axis=0), np.ones(window)/window, 'valid'),
        }

        fig, axs = plt.subplots(1, 6, figsize=[24, 6])

        axs[0].semilogy(runs.get('eval_calls'))
        axs[0].set_title('eval_calls')
        axs[1].semilogy(runs.get('f'))
        axs[1].set_title('f')
        axs[2].semilogy(runs.get('cr'))
        axs[2].set_title('cr')
        axs[3].semilogy(runs.get('front'))
        axs[3].set_title('front')
        axs[4].semilogy(runs.get('sr'))
        axs[4].set_title('sr')
        axs[5].semilogy(runs.get('sf'))
        axs[5].set_title('sf')

        fig.tight_layout()
        fig.savefig(Path('tmp', f'func_{func_num}.png'))
