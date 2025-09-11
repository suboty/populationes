#!/bin/bash

REPO_URL="https://github.com/VladimirStanovov/L-SRTDE_GNBG-24"
BRANCH="master"
FOLDER_IN_REPO="Results"
TARGET_DIR="Results"

load_original_results() {
  echo "Start loading of original L-SRTDE results"
  git clone --depth=1 --filter=blob:none --sparse -b "$BRANCH" "$REPO_URL" tmp_repo
  cd tmp_repo || exit 1
  git sparse-checkout set "$FOLDER_IN_REPO"
  mkdir -p "../$TARGET_DIR"
  cp -r "$FOLDER_IN_REPO"/* "../$TARGET_DIR"
  cd ..
  rm -rf tmp_repo
}

[ -d "$TARGET_DIR" ] || load_original_results
