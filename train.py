# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
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
import glob
import os
import shutil
import subprocess
import tempfile
from shlex import quote


ROOT = os.path.dirname(os.path.abspath(__file__))
FAST_ALIGN = ROOT + '/third-party/fast_align/build'
MOSES = ROOT + '/third-party/moses'
VECMAP = ROOT + '/third-party/vecmap'
PHRASE2VEC = ROOT + '/third-party/phrase2vec/word2vec'
TRAINING = ROOT + '/training'


def bash(command):
    subprocess.run(['bash', '-c', command])


def binarize(output_config, output_pt, lm_path, lm_order, phrase_table, reordering=None, prune=100):
    output_pt = os.path.abspath(output_pt)
    lm_path = os.path.abspath(lm_path)

    # Binarize
    reord_args = ' --lex-ro ' + quote(reordering) + ' --num-lex-scores 6' if reordering is not None else ''
    bash(quote(MOSES + '/scripts/generic/binarize4moses2.perl') +
         ' --phrase-table ' + quote(phrase_table) +
         ' --output-dir ' + quote(output_pt) +
         ' --num-scores 4' +
         ' --prune ' + str(prune) +
         reord_args)

    # Clean temporary files created by the binarization script
    for tmp in glob.glob(output_pt + '/../tmp.*'):
        shutil.rmtree(tmp)

    # Build configuration file
    with open(output_config, 'w') as f:
        print('[input-factors]', file=f)
        print('0', file=f)
        print('', file=f)
        print('[mapping]', file=f)
        print('0 T 0', file=f)
        print('', file=f)
        print('[distortion-limit]', file=f)
        print('6', file=f)
        print('', file=f)
        print('[feature]', file=f)
        print('UnknownWordPenalty', file=f)
        print('WordPenalty', file=f)
        print('PhrasePenalty', file=f)
        print('ProbingPT name=TranslationModel0 num-features=4' + 
              ' path=' + output_pt + ' input-factor=0 output-factor=0', file=f)
        if reordering is not None:
            print('LexicalReordering name=LexicalReordering0' +
                  ' num-features=6 type=wbe-msd-bidirectional-fe-allff' +
                  ' input-factor=0 output-factor=0 property-index=0', file=f)
        print('Distortion', file=f)
        print('KENLM name=LM0 factor=0 path=' + lm_path +
              ' order=' + str(lm_order), file=f)
        print('', file=f)
        print('[weight]', file=f)
        print('UnknownWordPenalty0= 1', file=f)
        print('WordPenalty0= -1', file=f)
        print('PhrasePenalty0= 0.2', file=f)
        print('TranslationModel0= 0.2 0.2 0.2 0.2', file=f)
        if reordering is not None:
            print('LexicalReordering0= 0.3 0.3 0.3 0.3 0.3 0.3', file=f)
        print('Distortion0= 0.3', file=f)
        print('LM0= 0.5', file=f)


def train_supervised(args, train_src, train_trg, dev_src, dev_trg, lm_path, lm_order, output_dir):
    lm_path = os.path.abspath(lm_path)
    output_dir = os.path.abspath(output_dir)
    tmp = args.tmp + '/train-supervised'
    os.mkdir(tmp)

    # Copy the corpus
    shutil.copy(train_src, tmp + '/corpus.src')
    shutil.copy(train_trg, tmp + '/corpus.trg')

    # Corpus cleaning
    bash(quote(MOSES + '/scripts/training/clean-corpus-n.perl') +
         ' ' + quote(tmp + '/corpus') + ' src trg' +
         ' ' + quote(tmp + '/clean') +
         ' ' + str(args.min_tokens) + ' ' + str(args.max_tokens))  # TODO Reusing min/max from step 1
    os.remove(tmp + '/corpus.src')
    os.remove(tmp + '/corpus.trg')

    # Merge both languages into a single file
    bash('paste -d " ||| " ' + quote(tmp + '/clean.src') +
         ' /dev/null /dev/null /dev/null /dev/null ' + quote(tmp + '/clean.trg') +
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
         ' -f src -e trg' +
         ' -alignment grow-diag-final-and' +
         ' -reordering msd-bidirectional-fe' +
         ' -max-phrase-length 5' +
         ' -temp-dir ' + quote(tmp + '/tmp') +
         ' -lm "0:{}:{}:8"'.format(lm_order, lm_path) +
         ' -first-step 4' +
         ' -score-options="-MinScore 2:0.0001"' +
         ' -cores ' + str(args.threads) +
         ' -parallel -sort-buffer-size 10G -sort-batch-size 253 -sort-compress gzip' +
         ' -sort-parallel ' + str(args.threads))
    shutil.move(tmp + '/phrase-table.gz', args.tmp)
    shutil.move(tmp + '/reordering-table.wbe-msd-bidirectional-fe.gz', args.tmp)
    shutil.rmtree(tmp)

    # Binarize model
    binarize(args.tmp + '/moses.ini',
             output_dir + '/probing-table',
             lm_path,
             lm_order,
             args.tmp + '/phrase-table.gz',
             args.tmp + '/reordering-table.wbe-msd-bidirectional-fe.gz',
             prune=args.pt_prune)
    # os.remove(args.tmp + '/phrase-table.gz')
    # os.remove(args.tmp + '/reordering-table.wbe-msd-bidirectional-fe.gz')
    shutil.move(args.tmp + '/phrase-table.gz', output_dir + '/phrase-table.gz')
    shutil.move(args.tmp + '/reordering-table.wbe-msd-bidirectional-fe.gz', output_dir + '/reordering-table.wbe-msd-bidirectional-fe.gz')

    # MERT
    bash(quote(MOSES + '/scripts/training/mert-moses.pl') +
         ' ' + quote(dev_src) +
         ' ' + quote(dev_trg) +
         ' ' + quote(MOSES + '/bin/moses2') +
         ' ' + quote(args.tmp + '/moses.ini') +
         ' --no-filter-phrase-table' +
         ' --mertdir ' + quote(MOSES + '/bin/') +
         ' --threads ' + str(args.threads) +
         ' --decoder-flags="-threads ' + str(args.threads) + '"' +
         ' --working-dir ' + quote(os.path.abspath(args.tmp + '/mert')))
    shutil.move(args.tmp + '/mert/moses.ini', output_dir + '/moses.ini')
    shutil.rmtree(args.tmp + '/mert')
    os.remove(args.tmp + '/moses.ini')


# Step 1: Corpus preprocessing
def preprocess(args):
    root = args.working + '/step1'
    os.mkdir(root)
    for part, corpus, lang in (('src', args.src, args.src_lang), ('trg', args.trg, args.trg_lang)):
        # Tokenize, deduplicate, clean by length, and shuffle
        bash('export LC_ALL=C;' +
             quote(MOSES + '/scripts/tokenizer/tokenizer.perl') +
             ' -l ' + quote(lang) + ' -threads ' + str(args.threads) +
             ' < ' + quote(corpus) +
             ' | sort -S 10G --batch-size 253 --compress-program gzip' +
             ' --parallel ' + str(args.threads) + ' -T ' + quote(args.tmp) +
             ' | uniq' + 
             ' | python3 ' + quote(TRAINING + '/clean-corpus.py') +
                 ' --min ' + str(args.min_tokens) +
                 ' --max ' + str(args.max_tokens) +
             ' | shuf'
             ' > ' + quote(args.tmp + '/full.tok'))

        # Train truecaser
        bash(quote(MOSES + '/scripts/recaser/train-truecaser.perl') +
             ' --model ' + quote(root + '/truecase-model.' + part) +
             ' --corpus ' + quote(args.tmp + '/full.tok'))

        # Truecase
        bash(quote(MOSES + '/scripts/recaser/truecase.perl') +
             ' --model ' + quote(root + '/truecase-model.' + part) +
             ' < ' + quote(args.tmp + '/full.tok') +
             ' > ' + quote(args.tmp + '/full.true'))

        # Split train/dev
        bash('head -' + str(args.dev_size) +
             ' < ' + quote(args.tmp + '/full.true') +
             ' > ' + quote(root + '/dev.true.' + part))
        bash('tail -n +' + str(args.dev_size + 1) +
             ' < ' + quote(args.tmp + '/full.true') +
             ' > ' + quote(root + '/train.true.' + part))

    # Remove temporary files
    os.remove(args.tmp + '/full.tok')
    os.remove(args.tmp + '/full.true')


# Step 2: Language model training
def train_lm(args):
    root = args.working + '/step2'
    os.mkdir(root)
    for part in ('src', 'trg'):
        bash(quote(MOSES + '/bin/lmplz') +
             ' -T ' + quote(args.tmp + '/lm') +
             ' -o ' + str(args.lm_order) +
             ' --prune ' + ' '.join(map(str, args.lm_prune)) +
             ' < ' + quote(args.working + '/step1/train.true.' + part) +
             ' > ' + quote(args.tmp + '/model.arpa'))
        bash(quote(MOSES + '/bin/build_binary') +
             ' ' + quote(args.tmp + '/model.arpa') +
             ' ' + quote(root + '/' + part + '.blm'))
        os.remove(args.tmp + '/model.arpa')


# Step 3: Train embeddings
def train_embeddings(args):
    root = args.working + '/step3'
    os.mkdir(root)
    for part in ('src', 'trg'):
        corpus = args.working + '/step1/train.true.' + part

        # Extract n-grams
        counts = []
        for i, cutoff in enumerate(args.vocab_cutoff):
            counts.append(quote(args.tmp + '/ngrams.' + str(i+1)))
            bash('python3 ' + quote(TRAINING + '/extract-ngrams.py') +
                 ' -i ' + quote(corpus) +
                 ' --min-order ' + str(i+1) +
                 ' --max-order ' + str(i+1) +
                 ' --min-count ' + str(args.vocab_min_count) +
                 ' | sort -nr' +
                 ' | head -' + str(cutoff) +
                 ' > ' + counts[-1])
        bash('cat ' + ' '.join(counts) + ' | cut -f2 > ' + quote(args.tmp + '/phrases.txt'))

        # Build standard word2vec vocabulary
        bash(quote(PHRASE2VEC) +
             ' -train ' + quote(corpus) +
             ' -min-count ' + str(args.vocab_min_count) +
             ' -save-vocab ' + quote(args.tmp + '/vocab-full.txt'))
        bash('head -' + str(args.vocab_cutoff[0]) +
             ' ' + quote(args.tmp + '/vocab-full.txt') +
             ' > ' + quote(args.tmp + '/vocab.txt'))

        # Train embeddings
        bash(quote(PHRASE2VEC) +
            ' -train ' + quote(corpus) +
            ' -read-vocab ' + quote(args.tmp + '/vocab.txt') +
            ' -phrases ' + quote(args.tmp + '/phrases.txt') +
            ' -cbow 0 -hs 0 -sample 0' +  # Fixed params
            ' -size ' + str(args.emb_size) +
            ' -window ' + str(args.emb_window) + 
            ' -negative ' + str(args.emb_negative) +
            ' -iter ' + str(args.emb_iter) +
            ' -threads ' + str(args.threads) +
            ' -output ' + quote(root + '/emb.' + part))

        # Clean temporary files
        for f in os.listdir(args.tmp):
            os.remove(os.path.join(args.tmp, f))


# Step 4: Map embeddings
# TODO Add CUDA support
def map_embeddings(args):
    root = args.working + '/step4'
    os.mkdir(root)
    bash('export OMP_NUM_THREADS=' + str(args.threads) + ';'
         ' python3 ' + quote(VECMAP + '/map_embeddings.py') +
         ' --unsupervised -v' +
         ' ' + quote(args.working + '/step3/emb.src') +
         ' ' + quote(args.working + '/step3/emb.trg') +
         ' ' + quote(root + '/emb.src') +
         ' ' + quote(root + '/emb.trg'))


# Step 5: Induce phrase-table
# TODO Add CUDA support
# TODO Add additional options
def induce_phrase_table(args):
    root = args.working + '/step5'
    os.mkdir(root)
    bash('export OMP_NUM_THREADS=' + str(args.threads) + ';'
         ' python3 ' + quote(TRAINING + '/induce-phrase-table.py') +
         ' --src ' + quote(args.working + '/step4/emb.src') +
         ' --trg ' + quote(args.working + '/step4/emb.trg') +
         ' --src2trg ' + quote(args.tmp + '/src2trg.phrase-table') +
         ' --trg2src ' + quote(args.tmp + '/trg2src.phrase-table'))
    for part in 'src2trg', 'trg2src':
        bash('export LC_ALL=C;' +
             ' sort -S 10G --batch-size 253 --compress-program gzip' +
             ' --parallel ' + str(args.threads) + ' -T ' + quote(args.tmp) +
             ' ' + quote(args.tmp + '/' + part + '.phrase-table') +
             ' | gzip > ' + quote(root + '/' + part + '.phrase-table.gz'))
        os.remove(args.tmp + '/' + part + '.phrase-table')


# Step 6: Build initial model
def build_initial_model(args):
    root = args.working + '/step6'
    os.mkdir(root)
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        part = src + '2' + trg
        binarize(root + '/' + part + '.moses.ini',
                 root + '/probing-table-' + part,
                 args.working + '/step2/' + trg + '.blm',
                 args.lm_order,
                 args.working + '/step5/' + part + '.phrase-table.gz',
                 prune=args.pt_prune)


# Step 7: Unsupervised tuning
def unsupervised_tuning(args):
    root = args.working + '/step7'
    os.mkdir(root)
    config = {('src', 'trg'): args.working + '/step6/src2trg.moses.ini',
              ('trg', 'src'): args.working + '/step6/trg2src.moses.ini'}
    for it in range(1, args.tuning_iter + 1):
        for src, trg in ('src', 'trg'), ('trg', 'src'):

            # Translate backwards
            bash(quote(MOSES + '/bin/moses2') +
                 ' -f ' + quote(config[(trg, src)]) + 
                 ' --threads ' + str(args.threads) +
                 ' < ' + quote(args.working + '/step1/dev.true.' + trg) +
                 ' > ' + quote(args.tmp + '/output.txt') +
                 ' 2> /dev/null')

            # MERT
            # TODO Should we start optimization from previous weights?
            bash(quote(MOSES + '/scripts/training/mert-moses.pl') +
                 ' ' + quote(args.tmp + '/output.txt') +
                 ' ' + quote(args.working + '/step1/dev.true.' + trg) +
                 ' ' + quote(MOSES + '/bin/moses2') +
                 ' ' + quote(config[(src, trg)]) +
                 ' --no-filter-phrase-table' +
                 ' --mertdir ' + quote(MOSES + '/bin/') +
                 ' --threads ' + str(args.threads) +
                 ' --decoder-flags="-threads ' + str(args.threads) + '"' +
                 ' --working-dir ' + quote(os.path.abspath(args.tmp + '/mert')))

            # Move tuned configuration file
            config[(src, trg)] = root + '/' + src + '2' + trg + '.it' + str(it) + '.moses.ini'
            shutil.move(args.tmp + '/mert/moses.ini', config[(src, trg)])

            # Remove temporary files
            shutil.rmtree(args.tmp + '/mert')
            os.remove(args.tmp + '/output.txt')

    shutil.copy(root + '/src2trg.it{0}.moses.ini'.format(args.tuning_iter), root + '/src2trg.moses.ini')
    shutil.copy(root + '/trg2src.it{0}.moses.ini'.format(args.tuning_iter), root + '/trg2src.moses.ini')


# Step 7 (alt): Supervised tuning
def supervised_tuning(args):
    root = args.working + '/step7'
    os.mkdir(root)
    for i, part, lang in (0, 'src', args.src_lang), (1, 'trg', args.trg_lang):
        bash(quote(MOSES + '/scripts/tokenizer/tokenizer.perl') +
             ' -l ' + quote(lang) + ' -threads ' + str(args.threads) +
             ' < ' + quote(args.supervised_tuning[i]) +
             ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl') +
             ' --model ' + quote(args.working + '/step1/truecase-model.' + part) +
             ' > ' + quote(args.tmp + '/dev.true.' + part))
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        bash(quote(MOSES + '/scripts/training/mert-moses.pl') +
             ' ' + quote(args.tmp + '/dev.true.' + src) +
             ' ' + quote(args.tmp + '/dev.true.' + trg) +
             ' ' + quote(MOSES + '/bin/moses2') +
             ' ' + args.working + '/step6/' + src + '2' + trg + '.moses.ini' +
             ' --no-filter-phrase-table' +
             ' --mertdir ' + quote(MOSES + '/bin/') +
             ' --threads ' + str(args.threads) +
             ' --decoder-flags="-threads ' + str(args.threads) + '"' +
             ' --working-dir ' + quote(os.path.abspath(args.tmp + '/mert')))
        shutil.move(args.tmp + '/mert/moses.ini', root + '/' + src + '2' + trg + '.moses.ini')
        shutil.rmtree(args.tmp + '/mert')
    os.remove(args.tmp + '/dev.true.src')
    os.remove(args.tmp + '/dev.true.trg')


# Step 8: Iterative backtranslation
def iterative_backtranslation(args):
    root = args.working + '/step8'
    os.mkdir(root)
    config = {('src', 'trg'): args.working + '/step7/src2trg.moses.ini',
              ('trg', 'src'): args.working + '/step7/trg2src.moses.ini'}
    for part in 'src', 'trg':
        bash('head -' + str(args.backtranslation_sentences) +
             ' ' + quote(args.working + '/step1/train.true.' + part) +
             ' > ' + quote(args.tmp + '/train.' + part))
    if args.supervised_tuning is not None:
        for i, part, lang in (0, 'src', args.src_lang), (1, 'trg', args.trg_lang):
            bash(quote(MOSES + '/scripts/tokenizer/tokenizer.perl') +
                 ' -l ' + quote(lang) + ' -threads ' + str(args.threads) +
                 ' < ' + quote(args.supervised_tuning[i]) +
                 ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl') +
                 ' --model ' + quote(args.working + '/step1/truecase-model.' + part) +
                 ' > ' + quote(args.tmp + '/dev.true.' + part))
    for it in range(1, args.backtranslation_iter + 1):
        for src, trg in ('src', 'trg'), ('trg', 'src'):
            # TODO Use cube pruning?
            bash(quote(MOSES + '/bin/moses2') +
                 ' -f ' + quote(config[(trg, src)]) + 
                 ' --threads ' + str(args.threads) +
                 ' < ' + quote(args.tmp + '/train.' + trg) +
                 ' > ' + quote(args.tmp + '/train.bt') +
                 ' 2> /dev/null')
            if args.supervised_tuning is not None:
                src_dev = args.tmp + '/dev.true.' + src
                trg_dev = args.tmp + '/dev.true.' + trg
            else:
                bash(quote(MOSES + '/bin/moses2') +
                     ' -f ' + quote(config[(trg, src)]) +
                     ' --threads ' + str(args.threads) +
                     ' < ' + quote(args.working + '/step1/dev.true.' + trg) +
                     ' > ' + quote(args.tmp + '/dev.bt') +
                     ' 2> /dev/null')
                src_dev = args.tmp + '/dev.bt'
                trg_dev = args.working + '/step1/dev.true.' + trg
            train_supervised(args,
                             args.tmp + '/train.bt',
                             args.tmp + '/train.' + trg,
                             src_dev,
                             trg_dev,
                             args.working + '/step2/' + trg + '.blm',
                             args.lm_order,
                             root + '/' + src + '2' + trg + '-it' + str(it))
            # os.remove(args.tmp + '/train.bt')
            shutil.move(args.tmp + '/train.bt', root + '/' + src + '2' + trg + '-it' + str(it) + '/train.bt')
            if args.supervised_tuning is None:
                os.remove(args.tmp + '/dev.bt')
            config[(src, trg)] = root + '/' + src + '2' + trg + '-it' + str(it) + '/moses.ini'

    shutil.copy(config[('src', 'trg')], args.working + '/src2trg.moses.ini')
    shutil.copy(config[('trg', 'src')], args.working + '/trg2src.moses.ini')
    # os.remove(args.tmp + '/train.src')
    # os.remove(args.tmp + '/train.trg')
    shutil.move(args.tmp + '/train.src', root + '/train.src')
    shutil.move(args.tmp + '/train.trg', root + '/train.trg')


def main():
    parser = argparse.ArgumentParser(description='Train an unsupervised SMT model')
    parser.add_argument('--src', metavar='PATH', required=True, help='Source language corpus')
    parser.add_argument('--trg', metavar='PATH', required=True, help='Target language corpus')
    parser.add_argument('--src-lang', metavar='STR', required=True, help='Source language code')
    parser.add_argument('--trg-lang', metavar='STR', required=True, help='Target language code')
    parser.add_argument('--from-step', metavar='N', type=int, default=1, help='Start at step N')
    parser.add_argument('--to-step', metavar='N', type=int, default=8, help='End at step N')
    parser.add_argument('--working', metavar='PATH', required=True, help='Working directory')
    parser.add_argument('--tmp', metavar='PATH', help='Temporary directory')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='Number of threads (defaults to 20)')

    parser.add_argument('--pt-prune', metavar='N', type=int, default=100, help='Phrase-table pruning (defaults to 100)')  # TODO Which group?

    preprocessing_group = parser.add_argument_group('Step 1', 'Corpus preprocessing')
    preprocessing_group.add_argument('--min-tokens', metavar='N', type=int, default=3, help='Remove sentences with less than N tokens (defaults to 3)')
    preprocessing_group.add_argument('--max-tokens', metavar='N', type=int, default=80, help='Remove sentences with more than N tokens (defaults to 80)')
    preprocessing_group.add_argument('--dev-size', metavar='N', type=int, default=10000, help='Number of sentences for tuning (defaults to 10000)')

    lm_group = parser.add_argument_group('Step 2', 'Language model training')
    lm_group.add_argument('--lm-order', metavar='N', type=int, default=5, help='Language model order (defaults to 5)')
    lm_group.add_argument('--lm-prune', metavar='N', type=int, nargs='+', default=[0, 0, 1], help='Language model pruning (defaults to 0 0 1)')

    phrase2vec_group = parser.add_argument_group('Step 3', 'Phrase embedding training')
    phrase2vec_group.add_argument('--vocab-cutoff', metavar='N', type=int, nargs='+', default=[200000, 400000, 400000], help='Vocabulary cut-off (defaults to 200000 400000 400000)')
    phrase2vec_group.add_argument('--vocab-min-count', metavar='N', type=int, default=10, help='Discard words with less than N occurrences (defaults to 10)')
    phrase2vec_group.add_argument('--emb-size', metavar='N', type=int, default=300, help='Dimensionality of the phrase embeddings (defaults to 300)')
    phrase2vec_group.add_argument('--emb-window', metavar='N', type=int, default=5, help='Max skip length between words (defauls to 5)')
    phrase2vec_group.add_argument('--emb-negative', metavar='N', type=int, default=10, help='Number of negative examples (defaults to 10)')
    phrase2vec_group.add_argument('--emb-iter', metavar='N', type=int, default=5, help='Number of training epochs (defaults to 5)')

    tuning_group = parser.add_argument_group('Step 7', 'Unsupervised tuning')
    tuning_group.add_argument('--tuning-iter', metavar='N', type=int, default=10, help='Number of unsupervised tuning iterations (defaults to 10)')
    tuning_group.add_argument('--supervised-tuning', metavar='PATH', nargs=2, help='Parallel corpus for supervised tuning (source/target)')  # TODO Also used for iterative backtranslation

    backtranslation_group = parser.add_argument_group('Step 8', 'Iterative backtranslation')
    backtranslation_group.add_argument('--backtranslation-iter', metavar='N', type=int, default=3, help='Number of backtranslation iterations (defaults to 3)')
    backtranslation_group.add_argument('--backtranslation-sentences', metavar='N', type=int, default=2000000, help='Number of sentences for training backtranslation (defaults to 2000000)')

    args = parser.parse_args()

    if args.tmp is None:
        args.tmp = args.working

    os.makedirs(args.working, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.tmp) as args.tmp:
        if args.from_step <= 1 <= args.to_step:
            preprocess(args)
        if args.from_step <= 2 <= args.to_step:
            train_lm(args)
        if args.from_step <= 3 <= args.to_step:
            train_embeddings(args)
        if args.from_step <= 4 <= args.to_step:
            map_embeddings(args)
        if args.from_step <= 5 <= args.to_step:
            induce_phrase_table(args)
        if args.from_step <= 6 <= args.to_step:
            build_initial_model(args)
        if args.from_step <= 7 <= args.to_step:
            if args.supervised_tuning is not None:
                supervised_tuning(args)
            else:
                unsupervised_tuning(args)
        if args.from_step <= 8 <= args.to_step:
            iterative_backtranslation(args)


if __name__ == '__main__':
    main()
