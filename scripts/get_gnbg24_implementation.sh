#!/bin/bash

current_path="l_srtde"

REPO_URL="https://github.com/VladimirStanovov/L-SRTDE_GNBG-24"
FILENAME="gnbg24.h"
BRANCH="master"

load_gnbg24_implementation() {
  echo "Start loading Vladimir Stanovov\`s GNBG 2024 code implementation"
  git clone --depth=1 --filter=blob:none --sparse -b "$BRANCH" "$REPO_URL" tmp_repo
  cd tmp_repo || exit 1
  git sparse-checkout init
  git sparse-checkout set "./$FILENAME"
  mv "$current_path/$FILENAME" "../$current_path/$FILENAME"
  cd ..
  rm -rf tmp_repo
}

load_gnbg24_implementation
