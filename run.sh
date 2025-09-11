#!/bin/bash

python_activate() {
  . "./.venv/bin/activate" || exit
}

python_activate

echo "Start Python implementation running"

read -r -p "Enter number of functions for Python implementation: " funcNum
read -r -p "Enter number of runs for Python implementation: " runNum

python3 run.py --runNum "$runNum" --funcNum "$funcNum"
echo "Python implementation finish, prepare results.."

cd l_srtde || exit
sh prepare_original_results.sh
python3 process_results.py --implementationFuncNum "$funcNum" --implementationRunNum "$runNum"

cd - || exit
echo "Done!"