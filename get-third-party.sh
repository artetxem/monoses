#!/bin/bash

CURRENT="$PWD"
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$ROOT/third-party"
cd "$ROOT/third-party"
git clone 'https://github.com/moses-smt/mosesdecoder.git' moses
cd moses
git checkout RELEASE-4.0
cd ..
git clone 'https://github.com/clab/fast_align.git'
git clone 'https://github.com/artetxem/phrase2vec.git'
git clone 'https://github.com/artetxem/vecmap.git'
cd "$CURRENT"