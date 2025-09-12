#!/bin/bash

current_path="$(pwd)"

read -r -p "Enter username: " username
read -r -p "Enter host: " host
read -r -p "Enter path to files: " resultPath

echo "Get original results"
scp "$username"@"$host":"$resultPath"/l_srtde/original_results.pdf "$current_path"/original_results.pdf
echo "Get python implementation results"
scp "$username"@"$host":"$resultPath"/l_srtde/python_implementation_results.pdf "$current_path"/python_implementation_results.pdf
echo "Get graphs"
scp "$username"@"$host":"$resultPath"/l_srtde/python_l_srtde_on_gnbg.png "$current_path"/python_l_srtde_on_gnbg.png