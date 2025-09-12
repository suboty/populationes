import subprocess
import argparse
from io import StringIO
from pathlib import Path

import numpy as np

def latex_to_pdf_subprocess(latex_file, output_dir="."):
    try:
        subprocess.run(["pdflatex", "-output-directory", output_dir, latex_file], check=True)
        print(f"Successfully converted {latex_file} to PDF in {output_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during LaTeX compilation: {e}")
    except FileNotFoundError:
        print(
            "Error: pdflatex command not found. "
            "Please ensure a LaTeX distribution is installed and in your PATH."
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='populationes-runs-results')
    parser.add_argument('--originalFuncNum', type=int, default=24)
    parser.add_argument('--originalRunNum', type=int, default=31)
    parser.add_argument('--implementationFuncNum', type=int, default=24)
    parser.add_argument('--implementationRunNum', type=int, default=31)
    args = parser.parse_args()

    path_to_python_implementation_results = "Results_Python_implementation"
    path_to_original_results = "Results"

    for path, prefix in [
        (path_to_python_implementation_results, 'python_implementation_'),
        (path_to_original_results, 'original_')
    ]:
        n_func = args.originalFuncNum if prefix == 'original_' else args.implementationFuncNum
        n_runs = args.originalRunNum + 1 if prefix == 'original_' else args.implementationRunNum
        all_res = np.zeros((n_func, n_runs, 1001))

        for func in range(n_func):
            with open(Path(path, f"L-SRTDE_GNBG_F{func+1}_D30.txt"), 'r') as f:
                all_res[func] = np.loadtxt(
                    StringIO(f.readline().rstrip()),
                )

        str1 = "Func & Absolute error & Required FEs to Acceptance Threshold & Success rate \\\\\n\\hline\n"
        for func in range(n_func):
            abs_err_mean = np.mean(all_res[func,:,-2])
            abs_err_std = np.std(all_res[func,:,-2])
            fe_mean = np.mean(all_res[func,:,-1])
            fe_std = np.std(all_res[func,:,-1])
            success = np.sum(all_res[func,:, -2] == 0) / n_runs

            str1 += f"F{func+1} & "
            str1 += f"${abs_err_mean:.6g} \\pm {abs_err_std:.6g}$ & "
            str1 += f"${fe_mean:.6g} \\pm {fe_std:.6g}$ & "
            str1 += f"{success:.6g} \\\\\n"

        with open(f'{prefix}results.tex', 'w') as file:
            file.write("""
            \\documentclass{article}
            \\usepackage{amsmath}
            \\begin{document}
            \\begin{table}[h!]
            \\centering
            \\begin{tabular}{l*{24}{c}}
            \\hline
            __RESULTS__
            \\hline
            \\end{tabular}
            \\end{table}
            \\end{document}
            """.replace('__RESULTS__', str1))

        latex_to_pdf_subprocess(f'{prefix}results.tex')
