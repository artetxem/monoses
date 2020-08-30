#!/bin/bash

CURRENT="$PWD"
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$ROOT/third-party"
cd "$ROOT/third-party"
git clone 'https://github.com/moses-smt/mosesdecoder.git' moses
cd moses
git checkout RELEASE-4.0
cd ..
git clone 'https://github.com/pytorch/fairseq.git'
cd fairseq
git checkout v0.6.0
cd ..
git clone 'https://github.com/clab/fast_align.git'
git clone 'https://github.com/artetxem/phrase2vec.git'
git clone 'https://github.com/artetxem/vecmap.git'
git clone 'https://github.com/rsennrich/subword-nmt.git'
git clone 'https://github.com/mjpost/sacrebleu.git'
cd "$CURRENT"