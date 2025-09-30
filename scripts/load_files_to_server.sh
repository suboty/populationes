#!/bin/bash

read -r -p "Enter username: " username
read -r -p "Enter host: " host
read -r -p "Enter path to files: " filesPath

echo "Load main running file"
scp ./run.py "$username"@"$host":"$filesPath"/run.py

echo "load GNBG 2024 data"
scp -r ./gnbg24 "$username"@"$host":"$filesPath"/gnbg24

echo "load L-SRTDE code"
scp -r ./l_srtde/*[!.pdf] ./l_srtde "$username"@"$host":"$filesPath"/l_srtde