#!/bin/bash

current_path='gnbg24'

ls mat || { echo "<mat> folder is not found!"; exit; }
for filename in *.mat; do
  mv "$current_path/$filename" "$current_path/mat/$filename"
done

ls txt || { echo "<txt> folder is not found!"; exit; }
for filename in *.txt; do
  mv "$current_path/$filename" "$current_path/txt/$filename"
done