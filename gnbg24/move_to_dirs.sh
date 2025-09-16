#!/bin/bash

ls mat || { echo "<mat> folder is not found!"; exit; }
for filename in *.mat; do
  mv "./$filename" "./mat/$filename"
done

ls txt || { echo "<txt> folder is not found!"; exit; }
for filename in *.txt; do
  mv "./$filename" "./txt/$filename"
done