#!/bin/bash

repoUrl="https://github.com/VladimirStanovov/L-SRTDE_GNBG-24"
filename="L-SRTDE.cpp"
branch="master"

load_original_code() {
  echo "Start loading Vladimir Stanovov's L-SRTDE original code"
  git clone --depth=1 --filter=blob:none --sparse -b "$branch" "$repoUrl" tmp_repo
  cd tmp_repo || exit 1
  git sparse-checkout init
  git sparse-checkout set "./$filename"
  mv "./$filename" "../$filename"
  cd ..
  rm -rf tmp_repo
}

prepare_original_code() {
  if sed --version >/dev/null 2>&1; then
    # Linux (GNU sed)
    sed -i 's|#include "gnbg24.h"|#include "../gnbg24/gnbg24.h"|g' "$filename"
  else
    # macOS / BSD sed
    sed -i '' 's|#include "gnbg24.h"|#include "../gnbg24/gnbg24.h"|g' "$filename"
  fi
}

load_original_code
prepare_original_code
