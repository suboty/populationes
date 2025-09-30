#!/bin/bash

python_activate() {
  . "./.venv/bin/activate" || exit
}

python_activate

echo "Start Python implementation running"

read -r -p "Enter number of functions for Python implementation: " funcNum
read -r -p "Enter number of runs for Python implementation: " runNum

read -r -p "Is need tmp values saving? (y/n): " tmpSaving
if [ "$tmpSaving" = "yes" ] || [ "$tmpSaving" = "y" ]; then
  tmpSaving='y'
fi

python3 run.py \
--runNum "$runNum" \
--funcNum "$funcNum" \
--isNeedTmpSaving "$tmpSaving"
echo "Python implementation finish, prepare results.."

cd l_srtde || exit
sh prepare_original_results.sh
python3 process_results.py \
--implementationFuncNum "$funcNum" \
--implementationRunNum "$runNum"

cd - || exit
echo "Done!"