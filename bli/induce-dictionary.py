# Copyright (C) 2019  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import numpy as np
import os
import shutil
import subprocess
import tempfile
from shlex import quote

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
PT2DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pt2dict.py')
THIRD_PARTY = os.path.abspath(os.environ['MONOSES_THIRD_PARTY']) if 'MONOSES_THIRD_PARTY' in os.environ else ROOT + '/third-party'
FAST_ALIGN = THIRD_PARTY + '/fast_align/build'
MOSES = THIRD_PARTY + '/moses'
TRAINING = ROOT + '/training'


def bash(command):
    subprocess.run(['bash', '-c', command])


def split_train_dev(args):
    root = args.working + '/step1'
    os.mkdir(root)
    for i, part in enumerate(['src', 'trg']):
        bash(quote(MOSES + '/scripts/tokenizer/escape-special-chars.perl') + ' < ' + quote(args.corpus[i]) +
             ' | shuf > ' + quote(args.tmp + '/full.true'))
        bash('head -' + str(args.dev_size) +
             ' < ' + quote(args.tmp + '/full.true') +
             ' > ' + quote(root + '/dev.true.' + part))
        bash('tail -n +' + str(args.dev_size + 1) +
             ' < ' + quote(args.tmp + '/full.true') +
             ' > ' + quote(root + '/train.true.' + part))
    os.remove(args.tmp + '/full.true')


def run_monoses(args, step):
    tmpdir = os.path.join(args.tmp, 'monoses')
    bash('python3 ' + quote(ROOT + '/train.py') +
         ' --src /dev/null --trg /dev/null --src-lang unk --trg-lang unk' +
         ' --no-levenshtein' +
         ' --working ' + quote(args.working) +
         ' --from-step ' + str(step) + ' --to-step ' + str(step) +
         ' --threads ' + str(args.threads) +
         ' --tmp ' + quote(tmpdir))


def ngram_embeddings(args):
    root = args.working + '/step4'
    os.mkdir(root)
    for i, part in enumerate(['src', 'trg']):
        corpus = args.working + '/step1/train.true.' + part

        # Extract n-grams
        counts = []
        for order in [2, 3]:  # TODO Hardcoded param (order)
            counts.append(quote(args.tmp + '/ngrams.' + str(order)))
            bash('python3 ' + quote(TRAINING + '/extract-ngrams.py') +
                 ' -i ' + quote(corpus) +
                 ' --min-order ' + str(order) +
                 ' --max-order ' + str(order) +
                 ' --min-count 5' +  # TODO Hardcoded param (min-count)
                 ' | sort -nr' +
                 ' | head -400000' +  # TODO Hardcoded param (cutoff)
                 ' > ' + counts[-1])
        bash('cat ' + ' '.join(counts) + ' | cut -f2 > ' + quote(args.tmp + '/phrases.txt'))
        for f in counts:
            os.remove(f)

        # Escape and read embeddings
        bash(quote(MOSES + '/scripts/tokenizer/escape-special-chars.perl') +
             ' < ' + quote(args.embeddings[i]) +
             ' > ' + quote(args.tmp + '/embeddings.txt'))
        with open(args.tmp + '/embeddings.txt', encoding='utf-8', errors='surrogateescape') as file:
            header = file.readline().split(' ')
            count = int(header[0])
            dim = int(header[1])
            words = []
            x = np.empty((count, dim), dtype='float32')
            for j in range(count):
                word, vec = file.readline().split(' ', 1)
                words.append(word)
                x[j] = np.fromstring(vec, sep=' ', dtype='float32')
        os.remove(args.tmp + '/embeddings.txt')

        # Length normalization
        norms = np.sqrt(np.sum(x ** 2, axis=1))
        norms[norms == 0] = 1
        x /= norms[:, np.newaxis]

        # Compute indices
        word2ind = {word: ind for ind, word in enumerate(words)}
        indices = [[j] for j in range(len(words))]
        with open(args.tmp + '/phrases.txt', encoding='utf-8', errors='surrogateescape') as file:
            for line in file:
                try:
                    ind = [word2ind[word] for word in line.strip().split()]
                    word = '&#32;'.join(line.strip().split())
                    indices.append(ind)
                    words.append(word)
                except Exception:
                    pass
                    # print('OOV: {}'.format(line.strip()))

        # Write embeddings
        with open(root + '/emb.' + part, mode='x', encoding='utf-8', errors='surrogateescape') as file:
            print('{} {}'.format(len(words), dim), file=file)
            for ind, word in zip(indices, words):
                emb = x[ind].sum(axis=0)
                emb /= np.sqrt(np.sum(emb**2))
                print(word + ' ' + ' '.join(['%.6g' % y for y in emb]), file=file)

        # Clean up
        del x
        os.remove(args.tmp + '/phrases.txt')


def unsupervised_tuning(args):
    if args.skip_tuning:
        root = args.working + '/step7'
        os.mkdir(root)
        shutil.copy(args.working + '/step6/src2trg.moses.ini', root + '/src2trg.moses.ini')
        shutil.copy(args.working + '/step6/trg2src.moses.ini', root + '/trg2src.moses.ini')
    else:
        run_monoses(args, 7)


def generate_synthetic_bitext(args):
    root = args.working + '/step8'
    os.mkdir(root)
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        direction = src + '2' + trg
        bash('head -' + str(args.bitext_sentences) +
             ' ' + quote(args.working + '/step1/train.true.' + src) +
             ' > ' + quote(root + '/bitext.' + direction + '.' + src))
        bash(quote(MOSES + '/bin/moses2') +
             ' -f ' + quote(args.working + '/step7/' + direction + '.moses.ini') +
             ' -search-algorithm 1 -cube-pruning-pop-limit {0} -s {0}'.format(args.cube_pruning_pop_limit) +
             ' --threads ' + str(args.threads) +
             ' < ' + quote(root + '/bitext.' + direction + '.' + src) +
             ' 2> /dev/null' +
             ' > ' + quote(root + '/bitext.' + direction + '.' + trg))


def build_phrase_table(args):
    root = args.working + '/step9'
    os.mkdir(root)
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        # We use back-translated data in the reverse direction (synthetic source and real target)
        direction = src + '2' + trg
        revdirection = trg + '2' + src

        # Create temporary working directory
        tmp = args.tmp + '/train-supervised'
        os.mkdir(tmp)

        # Corpus cleaning
        bash(quote(MOSES + '/scripts/training/clean-corpus-n.perl') +
             ' ' + quote(args.working + '/step8/bitext.' + revdirection) + ' ' + src + ' ' + trg +
             ' ' + quote(tmp + '/clean') +
             ' 3 80')  # TODO Hardcoded params

        # Shuffle
        bash('paste ' + quote(tmp + '/clean.' + src) + ' ' + quote(tmp + '/clean.' + trg) +
             ' | shuf > ' + quote(tmp + '/clean.shuf'))
        bash('cut -f1 ' + quote(tmp + '/clean.shuf') + ' > ' + quote(tmp + '/clean.' + src))
        bash('cut -f2 ' + quote(tmp + '/clean.shuf') + ' > ' + quote(tmp + '/clean.' + trg))
        os.remove(tmp + '/clean.shuf')

        # Merge both languages into a single file
        bash('paste -d " ||| " ' + quote(tmp + '/clean.' + src) +
             ' /dev/null /dev/null /dev/null /dev/null ' + quote(tmp + '/clean.' + trg) +
             ' > ' + quote(tmp + '/clean.txt'))

        # Align
        bash(quote(FAST_ALIGN + '/fast_align') +
             ' -i ' + quote(tmp + '/clean.txt') + ' -d -o -v' +
             ' > ' + quote(tmp + '/forward.align'))
        bash(quote(FAST_ALIGN + '/fast_align') +
             ' -i ' + quote(tmp + '/clean.txt') + ' -d -o -v -r' +
             ' > ' + quote(tmp + '/reverse.align'))
        os.remove(tmp + '/clean.txt')

        # Symmetrization
        bash(quote(FAST_ALIGN + '/atools') +
             ' -i ' + quote(tmp + '/forward.align') +
             ' -j ' + quote(tmp + '/reverse.align') +
             ' -c grow-diag-final-and' +
             ' > ' + quote(tmp + '/aligned.grow-diag-final-and'))
        os.remove(tmp + '/forward.align')
        os.remove(tmp + '/reverse.align')

        # Build model
        bash(quote(MOSES + '/scripts/training/train-model.perl') +
             ' -model-dir ' + quote(tmp) +
             ' -corpus ' + quote(tmp) + '/clean' +
             ' -f ' + src + ' -e ' + trg +
             ' -alignment grow-diag-final-and' +
             ' -max-phrase-length 3' +  # TODO Hardcoded param (using 3 instead of standard 5)
             ' -temp-dir ' + quote(tmp + '/tmp') +
             ' -first-step 4' +
             ' -last-step 6' +
             ' -score-options="-MinScore 2:0.0001"' +  # TODO Hardcoded param
             ' -cores ' + str(args.threads) +
             ' -parallel -sort-buffer-size 10G -sort-batch-size 253 -sort-compress gzip' +
             ' -sort-parallel ' + str(args.threads))

        # Move the phrase-table and remove the temporary working directory
        shutil.move(tmp + '/phrase-table.gz', root + '/' + direction + '.phrase-table.gz')
        shutil.rmtree(tmp)


def induce_dictionary(args):
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        bash('zcat ' + quote(args.working + '/step9/' + src + '2' + trg + '.phrase-table.gz') +
                ' | python3 ' + quote(PT2DICT) + ' -f ' + str(args.feature) + (' -r' if args.reverse else '') + (' --phrases' if args.phrases else '') +
                ' | ' + quote(MOSES + '/scripts/tokenizer/deescape-special-chars.perl') +
                ' | LC_ALL=C sort -k1,1 -k3,3gr' +
                ' > ' + quote(args.working + '/' + (trg + '2' + src if args.reverse else src + '2' + trg) + '.dic'))


def main():
    parser = argparse.ArgumentParser(description='Induce a dictionary from cross-lingual word embeddings')
    parser.add_argument('--corpus', nargs=2, metavar='PATH', required=True, help='Tokenized monolingual corpora (src, trg)')
    parser.add_argument('--embeddings', nargs=2, metavar='PATH', required=True, help='Cross-lingual embeddings (src, trg)')
    parser.add_argument('--working', metavar='PATH', required=True, help='Working directory')
    parser.add_argument('--tmp', metavar='PATH', help='Temporary directory')
    parser.add_argument('--from-step', metavar='N', type=int, default=1, help='Start at step N')
    parser.add_argument('--to-step', metavar='N', type=int, default=10, help='End at step N')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='Number of threads (defaults to 20)')
    parser.add_argument('--dev-size', metavar='N', type=int, default=2000, help='Number of sentences for tuning (defaults to 2000)')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip unsupervised tuning (use default Moses weights)')
    parser.add_argument('--cube-pruning-pop-limit', metavar='N', type=int, default=1000, help='Cube pruning pop limit for fast decoding (defaults to 1000)')
    parser.add_argument('--bitext-sentences', metavar='N', type=int, default=10000000, help='Number of sentences for synthetic bitext generation (defaults to 10000000)')
    parser.add_argument('-f', '--feature', default=2, type=int, help='The phrase-table index of the feature to use for scoring (defaults to 2)')
    parser.add_argument('-r', '--reverse', action='store_true', help='Use the phrase-table in the reverse direction (trg -> src) to extract the dictionary')
    parser.add_argument('-p', '--phrases', action='store_true', help='Include phrases in the induced dictionary')
    args = parser.parse_args()

    if args.tmp is None:
        args.tmp = args.working

    os.makedirs(args.working, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.tmp) as args.tmp:
        if args.from_step <= 1 <= args.to_step:
            split_train_dev(args)
        if args.from_step <= 2 <= args.to_step:
            run_monoses(args, 2)  # LM training
        if args.from_step <= 3 <= args.to_step:
            pass
        if args.from_step <= 4 <= args.to_step:
            ngram_embeddings(args)
        if args.from_step <= 5 <= args.to_step:
            run_monoses(args, 5)  # Phrase-table induction
        if args.from_step <= 6 <= args.to_step:
            run_monoses(args, 6)  # Phrase-table binarization
        if args.from_step <= 7 <= args.to_step:
            unsupervised_tuning(args)
        if args.from_step <= 8 <= args.to_step:
            generate_synthetic_bitext(args)
        if args.from_step <= 9 <= args.to_step:
            build_phrase_table(args)
        if args.from_step <= 10 <= args.to_step:
            induce_dictionary(args)


if __name__ == '__main__':
    main()
