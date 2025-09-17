#!/bin/bash

current_path="$(pwd)"

read -r -p "Enter username: " username
read -r -p "Enter host: " host
read -r -p "Enter path to files: " filesPath

echo "Load main running file"
scp "$current_path"/run.py "$username"@"$host":"$filesPath"/run.py

echo "load L-SRTDE code"
scp -r ./l_srtde/*[!.pdf] "$current_path"/l_srtde "$username"@"$host":"$filesPath"/l_srtde