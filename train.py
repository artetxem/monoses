# Copyright (C) 2018-2020  Mikel Artetxe <artetxem@gmail.com>
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
import gzip
import os
import shutil
import subprocess
import tempfile
from shlex import quote


ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.abspath(os.environ['MONOSES_THIRD_PARTY']) if 'MONOSES_THIRD_PARTY' in os.environ else ROOT + '/third-party'
FAST_ALIGN = THIRD_PARTY + '/fast_align/build'
MOSES = THIRD_PARTY + '/moses'
FAIRSEQ = THIRD_PARTY + '/fairseq'
VECMAP = THIRD_PARTY + '/vecmap'
SUBWORD_NMT = THIRD_PARTY + '/subword-nmt'
PHRASE2VEC = THIRD_PARTY + '/phrase2vec/word2vec'
TRAINING = ROOT + '/training'


def bash(command):
    subprocess.run(['bash', '-c', command])


def count_lines(path):
    return int(subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE).stdout.decode('utf-8').strip().split()[0])


def binarize(output_config, output_pt, lm_path, lm_order, phrase_table, reordering=None, pt_scores=4, prune=100):
    output_pt = os.path.abspath(output_pt)
    lm_path = os.path.abspath(lm_path)

    # Binarize
    reord_args = ' --lex-ro ' + quote(reordering) + ' --num-lex-scores 6' if reordering is not None else ''
    bash(quote(MOSES + '/scripts/generic/binarize4moses2.perl') +
         ' --phrase-table ' + quote(phrase_table) +
         ' --output-dir ' + quote(output_pt) +
         ' --num-scores ' + str(pt_scores) +
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
        print('ProbingPT name=TranslationModel0 num-features=' + str(pt_scores) +
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
        print('TranslationModel0=' + (' 0.2'*pt_scores), file=f)
        if reordering is not None:
            print('LexicalReordering0= 0.3 0.3 0.3 0.3 0.3 0.3', file=f)
        print('Distortion0= 0.3', file=f)
        print('LM0= 0.5', file=f)


def tune(args, input_src2trg, input_trg2src, output_src2trg, output_trg2src):
    if args.supervised_tuning is not None:
        for i, part, lang in (0, 'src', args.src_lang), (1, 'trg', args.trg_lang):
            bash('cat ' + quote(args.supervised_tuning[i]) +
                 ' | ' + tokenize_command(args, lang) +
                 ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl') +
                 ' --model ' + quote(args.working + '/step1/truecase-model.' + part) +
                 ' > ' + quote(args.tmp + '/dev.true.' + part))
    else:
        shutil.copy(args.working + '/step1/dev.true.src', args.tmp + '/dev.true.src')
        shutil.copy(args.working + '/step1/dev.true.trg', args.tmp + '/dev.true.trg')
    bash('python3 ' + quote(ROOT + '/training/tuning/tune.py') +
         ' --dev ' + quote(args.tmp + '/dev.true.src') + ' ' + quote(args.tmp + '/dev.true.trg') +
         ' --moses ' + quote(MOSES) +
         ' --input ' + quote(input_src2trg) + ' ' + quote(input_trg2src) +
         ' --output ' + quote(output_src2trg) + ' ' + quote(output_trg2src) +
         ' --threads ' + str(args.threads) +
         ' --cube-pruning-pop-limit ' + str(args.cube_pruning_pop_limit) +
         ' --iterations {}'.format(args.tuning_iter) +
         (' --length-init' if args.length_init else '') +
         ('' if args.supervised_tuning is None else ' --supervised'))
    os.remove(args.tmp + '/dev.true.src')
    os.remove(args.tmp + '/dev.true.trg')


def tokenize_command(args, lang):
    return quote(MOSES + '/scripts/tokenizer/normalize-punctuation.perl') + ' -l ' + quote(lang) + \
           ' | ' + quote(MOSES + '/scripts/tokenizer/remove-non-printing-char.perl') + \
           ' | ' + quote(MOSES + '/scripts/tokenizer/tokenizer.perl') + ' -q -a -l ' + quote(lang) + ' -threads ' + str(args.threads)


# Step 1: Corpus preprocessing
def preprocess(args):
    root = args.working + '/step1'
    os.mkdir(root)
    for part, corpus, lang in (('src', args.src, args.src_lang), ('trg', args.trg, args.trg_lang)):
        # Tokenize, deduplicate, clean by length, and shuffle
        bash('export LC_ALL=C;' +
             ' cat ' + quote(corpus) +
             ' | ' + tokenize_command(args, lang) +
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
         ' --' + args.vecmap_mode + ' -v' +
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
             ('' if args.no_levenshtein else (' |  python3 ' + quote(TRAINING + '/add-levenshtein.py'))) +
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
                 pt_scores= 4 if args.no_levenshtein else 6,
                 prune=args.pt_prune)

# Step 7: Tuning
def tuning(args):
    root = args.working + '/step7'
    os.mkdir(root)
    tune(args, args.working + '/step6/src2trg.moses.ini', args.working + '/step6/trg2src.moses.ini',
         root + '/src2trg.moses.ini', root + '/trg2src.moses.ini')


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
            bash('cat ' + quote(args.supervised_tuning[i]) +
                 ' | ' + tokenize_command(args, lang) +
                 ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl') +
                 ' --model ' + quote(args.working + '/step1/truecase-model.' + part) +
                 ' > ' + quote(args.tmp + '/dev.true.' + part))

    for it in range(1, args.backtranslation_iter + 1):
        for src, trg in ('src', 'trg'), ('trg', 'src'):
            # Backtranslation
            bash(quote(MOSES + '/bin/moses2') +
                 ' -f ' + quote(config[(trg, src)]) +
                 ' -search-algorithm 1 -cube-pruning-pop-limit {0} -s {0}'.format(args.cube_pruning_pop_limit) +
                 ' --threads ' + str(args.threads) +
                 ' < ' + quote(args.tmp + '/train.' + trg) +
                 ' > ' + quote(args.tmp + '/train.bt') +
                 ' 2> /dev/null')

            output_dir = root + '/' + src + '2' + trg + '-it' + str(it)
            tmp = args.tmp + '/train-supervised'
            os.mkdir(output_dir)
            os.mkdir(tmp)

            # Corpus cleaning
            bash(quote(MOSES + '/scripts/training/clean-corpus-n.perl') +
                 ' ' + quote(args.tmp + '/train') + ' bt ' + trg +
                 ' ' + quote(tmp + '/clean') +
                 ' ' + str(args.min_tokens) + ' ' + str(args.max_tokens))  # TODO Reusing min/max from step 1

            # Merge both languages into a single file
            bash('paste -d " ||| " ' + quote(tmp + '/clean.bt') +
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
                 ' -f bt -e ' + trg +
                 ' -alignment grow-diag-final-and' +
                 ' -reordering msd-bidirectional-fe' +
                 ' -max-phrase-length 5' +
                 ' -temp-dir ' + quote(tmp + '/tmp') +
                 ' -lm "0:{}:{}:8"'.format(args.lm_order, os.path.abspath(args.working + '/step2/' + trg + '.blm')) +
                 ' -first-step 4' +
                 ' -score-options="-MinScore 2:0.0001"' +
                 ' -cores ' + str(args.threads) +
                 ' -parallel -sort-buffer-size 10G -sort-batch-size 253 -sort-compress gzip' +
                 ' -sort-parallel ' + str(args.threads))
            shutil.move(tmp + '/phrase-table.gz', args.tmp + '/' + src + '2' + trg + '.phrase-table.gz')
            shutil.move(tmp + '/reordering-table.wbe-msd-bidirectional-fe.gz', output_dir)
            shutil.rmtree(tmp)

            # os.remove(args.tmp + '/train.bt')
            shutil.move(args.tmp + '/train.bt', output_dir + '/train.bt')

        for src, trg in ('src', 'trg'), ('trg', 'src'):
            output_dir = root + '/' + src + '2' + trg + '-it' + str(it)

            # Phrase table format
            # SRC TRG ||| TRG2SRC_PROB TRG2SRC_LEX SRC2TRG_PROB SRC2TRG_LEX ||| TRG_COUNT SRC_COUNT JOINT_COUNT

            # Merge phrase-tables
            phrase2scores = {}
            with gzip.open(args.tmp + '/' + trg + '2' + src + '.phrase-table.gz', mode='rt', encoding='utf-8', errors='surrogateescape') as f:
                for line in f:
                    cols = line.split('|||')
                    phrase2scores[(cols[1].strip(), cols[0].strip())] = cols[2].strip()
            with gzip.open(args.tmp + '/' + src + '2' + trg + '.phrase-table.gz', mode='rt', encoding='utf-8', errors='surrogateescape') as fin:
                with gzip.open(output_dir + '/phrase-table.gz', mode='xt', encoding='utf-8', errors='surrogateescape') as fout:
                    for line in fin:
                        cols = line.split('|||')
                        scores = phrase2scores.get((cols[0].strip(), cols[1].strip()))
                        if scores is not None:
                            cols[2] = ' ' + ' '.join(scores.strip().split()[2:] + cols[2].strip().split()[2:]) + ' '
                            print('|||'.join(cols), end='', file=fout)
            del phrase2scores

            # Binarize model
            binarize(output_dir + '/default.moses.ini',
                     output_dir + '/probing-table',
                     os.path.abspath(args.working + '/step2/' + trg + '.blm'),
                     args.lm_order,
                     output_dir + '/phrase-table.gz',
                     output_dir + '/reordering-table.wbe-msd-bidirectional-fe.gz',
                     prune=args.pt_prune)

        if not args.no_backtranslation_tuning:
            tune(args,
                 '{}/src2trg-it{}/default.moses.ini'.format(root, it),
                 '{}/trg2src-it{}/default.moses.ini'.format(root, it),
                 '{}/src2trg-it{}/moses.ini'.format(root, it),
                 '{}/trg2src-it{}/moses.ini'.format(root, it))
        else:
            shutil.copy('{}/src2trg-it{}/default.moses.ini'.format(root, it), '{}/src2trg-it{}/moses.ini'.format(root, it))
            shutil.copy('{}/trg2src-it{}/default.moses.ini'.format(root, it), '{}/trg2src-it{}/moses.ini'.format(root, it))

        config[('src', 'trg')] = '{}/src2trg-it{}/moses.ini'.format(root, it)
        config[('trg', 'src')] = '{}/trg2src-it{}/moses.ini'.format(root, it)

        # os.remove(args.tmp + '/src2trg.phrase-table.gz')
        # os.remove(args.tmp + '/trg2src.phrase-table.gz')
        shutil.move(args.tmp + '/src2trg.phrase-table.gz', root + '/src2trg-' + str(it) + '.phrase-table.gz')
        shutil.move(args.tmp + '/trg2src.phrase-table.gz', root + '/trg2src-' + str(it) + '.phrase-table.gz')

    # TODO Tuning in the last iteration is unnecessary

    if args.supervised_tuning is None:  # Use default weights in the final model
        shutil.copy('{}/src2trg-it{}/default.moses.ini'.format(root, args.backtranslation_iter), root + '/src2trg.moses.ini')
        shutil.copy('{}/trg2src-it{}/default.moses.ini'.format(root, args.backtranslation_iter), root + '/trg2src.moses.ini')
    else:
        shutil.copy('{}/src2trg-it{}/moses.ini'.format(root, args.backtranslation_iter), root + '/src2trg.moses.ini')
        shutil.copy('{}/trg2src-it{}/moses.ini'.format(root, args.backtranslation_iter), root + '/trg2src.moses.ini')
    # os.remove(args.tmp + '/train.src')
    # os.remove(args.tmp + '/train.trg')
    shutil.move(args.tmp + '/train.src', root + '/train.src')
    shutil.move(args.tmp + '/train.trg', root + '/train.trg')


# Step 9: Synthetic parallel corpus generation
def generate_bitext(args):
    root = args.working + '/step9'
    os.mkdir(root)

    # Concatenate and shuffle both corpora oversampling the smallest one, and learn BPE on it
    src_lines = count_lines(args.working + '/step1/train.true.src')
    trg_lines = count_lines(args.working + '/step1/train.true.trg')
    if src_lines > trg_lines:
        max_lines, max_corpus = src_lines, args.working + '/step1/train.true.src'
        min_lines, min_corpus = trg_lines, args.working + '/step1/train.true.trg'
    else:
        max_lines, max_corpus = trg_lines, args.working + '/step1/train.true.trg'
        min_lines, min_corpus = src_lines, args.working + '/step1/train.true.src'
    bash('cat ' + quote(min_corpus) +
         ' | shuf' +
         ' | head -' + str(max_lines % min_lines) +
         ' | cat - ' + ' '.join([quote(min_corpus)] * (max_lines // min_lines)) + ' ' + quote(max_corpus) +
         ' | shuf' +
         ' > ' + quote(args.tmp + '/all.txt'))
    bash('python3 ' + quote(SUBWORD_NMT + '/subword_nmt/learn_bpe.py') + ' -s ' + str(args.bpe_tokens) +
         ' < ' + quote(args.tmp + '/all.txt') +
         ' > ' + quote(root + '/bpe.codes'))
    os.remove(args.tmp + '/all.txt')

    # Backtranslate and apply BPE
    for src, trg in ('src', 'trg'), ('trg', 'src'):
        bash('head -' + str(args.bitext_sentences) + ' ' + quote(args.working + '/step1/train.true.' + src) +
             ' | ' + quote(MOSES + '/bin/moses2') +
             ' -f ' + quote(args.working + '/step8/' + src + '2' + trg + '.moses.ini') +
             ' -search-algorithm 1 -cube-pruning-pop-limit {0} -s {0}'.format(args.cube_pruning_pop_limit) +
             ' --threads ' + str(args.threads) +
             ' 2> /dev/null' +
             ' | python3 ' + quote(SUBWORD_NMT + '/subword_nmt/apply_bpe.py') +
             ' -c ' + quote(root + '/bpe.codes') +
             ' > ' + quote(root + '/train.' + trg + '2' + src + '.' + trg))
        bash('python3 ' + quote(SUBWORD_NMT + '/subword_nmt/apply_bpe.py') +
             ' -c ' + quote(root + '/bpe.codes') +
             ' < ' + quote(args.working + '/step1/train.true.' + src) +
             ' > ' + quote(root + '/train.' + trg + '2' + src + '.' + src))

    # Extract vocabulary
    bash('cat ' + quote(root + '/train.trg2src.src') + ' ' + quote(root + '/train.src2trg.trg') +
         ' | python3 ' + quote(SUBWORD_NMT + '/subword_nmt/get_vocab.py') +
         ' > ' + quote(root + '/vocab.txt'))


# Step 10: NMT training
def train_nmt(args):
    root = args.working + '/step10'
    os.mkdir(root)

    src_reader = open(args.working + '/step9/train.trg2src.src', encoding='utf-8', errors='surrogateescape')
    trg_reader = open(args.working + '/step9/train.src2trg.trg', encoding='utf-8', errors='surrogateescape')
    src2trg_src_reader = open(args.working + '/step9/train.src2trg.src', encoding='utf-8', errors='surrogateescape')
    src2trg_trg_reader = open(args.working + '/step9/train.src2trg.trg', encoding='utf-8', errors='surrogateescape')
    trg2src_src_reader = open(args.working + '/step9/train.trg2src.src', encoding='utf-8', errors='surrogateescape')
    trg2src_trg_reader = open(args.working + '/step9/train.trg2src.trg', encoding='utf-8', errors='surrogateescape')

    # Skip initial lines for the monolingual readers
    for i in range(count_lines(args.working + '/step9/train.trg2src.trg')):
        src_reader.readline()
    for i in range(count_lines(args.working + '/step9/train.src2trg.src')):
        trg_reader.readline()
    
    bash('echo . > ' + quote(args.tmp + '/dummy.src'))
    bash('echo . > ' + quote(args.tmp + '/dummy.trg'))
    for it in range(1, args.nmt_iter + 1):
        print('IT {}'.format(it))
        for src, trg, smt_src, smt_trg, mono_trg in ('src', 'trg', src2trg_src_reader, src2trg_trg_reader, trg_reader), \
                                                    ('trg', 'src', trg2src_trg_reader, trg2src_src_reader, src_reader):
            
            # Compute SMT/NMT ratio
            nmt_percentage = min(1.0, (it - 1) / (args.nmt_transition_iter - 1))
            nmt_sentences_per_gpu = int(args.nmt_sentences_per_iter * nmt_percentage) // len(args.nmt_gpus)
            smt_sentences = args.nmt_sentences_per_iter - nmt_sentences_per_gpu * len(args.nmt_gpus)

            # Build (copy) SMT training
            with open(args.tmp + '/train.' + src, mode='w', encoding='utf-8', errors='surrogateescape') as fsrc:
                with open(args.tmp + '/train.' + trg, mode='w', encoding='utf-8', errors='surrogateescape') as ftrg:
                    count = 0
                    while count < smt_sentences:
                        srcsent = smt_src.readline()
                        trgsent = smt_trg.readline()
                        if srcsent == '' or trgsent == '':
                            smt_src.seek(0)
                            smt_trg.seek(0)
                        else:
                            count += 1
                            print(srcsent, end='', file=fsrc)
                            print(trgsent, end='', file=ftrg)

            # Build NMT training
            if nmt_sentences_per_gpu > 0:
                command = ''
                for i, gpu in enumerate(args.nmt_gpus):
                    with open(args.tmp + '/mono.' + str(gpu), mode='w', encoding='utf-8', errors='surrogateescape') as f:
                        count = 0
                        while count < nmt_sentences_per_gpu:
                            sent = mono_trg.readline()
                            if sent == '':
                                mono_trg.seek(0)
                            else:
                                count += 1
                                print(sent, end='', file=f)
                    command += 'cat ' + quote(args.tmp + '/mono.' + str(gpu))
                    command += ' | CUDA_VISIBLE_DEVICES=' + str(gpu)
                    command += ' python3 ' + quote(FAIRSEQ + '/interactive.py') + ' ' + quote(args.tmp + '/' + trg + '2' + src + '.data.bin')
                    command += ' --path ' + quote(args.tmp + '/' + trg + '2' + src + '/checkpoint_last.pt')
                    command += ' --beam 1'
                    if i % 2 == 0:  # TODO Assuming that the number of GPUs is even
                        command += ' --sampling'
                    command += ' --max-tokens 10000'
                    command += ' --max-len-a 2'
                    command += ' --max-len-b 5'
                    command += ' --buffer-size ' + str(nmt_sentences_per_gpu)
                    if args.nmt_fp16:
                        command += ' --fp16'
                    command += ' | grep -P \'^H\t\''
                    command += ' | cut -f3'
                    command += ' > ' + quote(args.tmp + '/bt.' + str(gpu))
                    command += ' &\n'
                bash(command + 'wait')
                for gpu in args.nmt_gpus:
                    bash('cat ' + quote(args.tmp + '/bt.' + str(gpu)) + ' >> ' + quote(args.tmp + '/train.' + src))
                    bash('cat ' + quote(args.tmp + '/mono.' + str(gpu)) + ' >> ' + quote(args.tmp + '/train.' + trg))
                    os.remove(args.tmp + '/bt.' + str(gpu))
                    os.remove(args.tmp + '/mono.' + str(gpu))

            # Binarize training data
            bash('python3 ' + quote(FAIRSEQ + '/preprocess.py') +
                ' --source-lang ' + src +
                ' --target-lang ' + trg +
                ' --trainpref ' + quote(args.tmp + '/train') +
                ' --validpref ' + quote(args.tmp + '/dummy') +
                ' --destdir ' + quote(args.tmp + '/' + src + '2' + trg + '.data.bin') +
                ' --srcdict ' + quote(args.working + '/step9/vocab.txt') +
                ' --tgtdict ' + quote(args.working + '/step9/vocab.txt') +
                ' --workers ' + str(args.threads))
            os.remove(args.tmp + '/train.src')
            os.remove(args.tmp + '/train.trg')

            # Train NMT
            bash('CUDA_VISIBLE_DEVICES=' + ','.join([str(gpu) for gpu in args.nmt_gpus]) +
                 ' python3 ' + quote(FAIRSEQ + '/train.py') + ' ' + quote(args.tmp + '/' + src + '2' + trg + '.data.bin') +
                 ' --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings' +
                 ' --optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0' +
                 ' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000' +
                 ' --lr 0.0005 --min-lr 1e-09' +
                 ' --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1' +
                 ' --max-tokens 2500' +
                 ' --update-freq ' +  str(args.nmt_cumul) +
                 ' --save-dir ' + quote(args.tmp + '/' + src + '2' + trg) +
                 ' --save-interval 1 --save-interval-updates 0 --no-epoch-checkpoints' +
                 ' --max-epoch 1' +
                 ' --log-format simple --log-interval 100' +
                 (' --fp16' if args.nmt_fp16 else ''))

            # Reset training iterator
            bash('CUDA_VISIBLE_DEVICES=' + str(args.nmt_gpus[0]) +
                 ' python3 ' + quote(TRAINING + '/reset-fairseq-iterator.py') +
                 ' ' + quote(args.tmp + '/' + src + '2' + trg + '/checkpoint_last.pt'))
            
            # Save checkpoint
            if it % args.nmt_save_interval == 0:
                shutil.copy(args.tmp + '/' + src + '2' + trg + '/checkpoint_last.pt', root + '/' + src + '2' + trg + '.' + str(it) + '.pt')
    
    # Save final checkpoint
    shutil.copy(args.tmp + '/src2trg/checkpoint_last.pt', root + '/src2trg.pt')
    shutil.copy(args.tmp + '/trg2src/checkpoint_last.pt', root + '/trg2src.pt')

    # Save dictionaries
    os.mkdir(root + '/data.bin')
    shutil.copy(args.tmp + '/src2trg.data.bin/dict.src.txt', root + '/data.bin/dict.src.txt')
    shutil.copy(args.tmp + '/src2trg.data.bin/dict.trg.txt', root + '/data.bin/dict.trg.txt')

    # Cleaning
    os.remove(args.tmp + '/dummy.src')
    os.remove(args.tmp + '/dummy.trg')
    shutil.rmtree(args.tmp + '/src2trg')
    shutil.rmtree(args.tmp + '/trg2src')
    shutil.rmtree(args.tmp + '/src2trg.data.bin')
    shutil.rmtree(args.tmp + '/trg2src.data.bin')


def main():
    parser = argparse.ArgumentParser(description='Train an unsupervised SMT model')
    parser.add_argument('--src', metavar='PATH', required=True, help='Source language corpus')
    parser.add_argument('--trg', metavar='PATH', required=True, help='Target language corpus')
    parser.add_argument('--src-lang', metavar='STR', required=True, help='Source language code')
    parser.add_argument('--trg-lang', metavar='STR', required=True, help='Target language code')
    parser.add_argument('--from-step', metavar='N', type=int, default=1, help='Start at step N')
    parser.add_argument('--to-step', metavar='N', type=int, default=10, help='End at step N')
    parser.add_argument('--working', metavar='PATH', required=True, help='Working directory')
    parser.add_argument('--tmp', metavar='PATH', help='Temporary directory')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='Number of threads (defaults to 20)')

    parser.add_argument('--pt-prune', metavar='N', type=int, default=100, help='Phrase-table pruning (defaults to 100)')  # TODO Which group?
    parser.add_argument('--cube-pruning-pop-limit', metavar='N', type=int, default=1000, help='Cube pruning pop limit for fast decoding (defaults to 1000)')  # TODO Which group?

    preprocessing_group = parser.add_argument_group('Step 1', 'Corpus preprocessing')
    preprocessing_group.add_argument('--min-tokens', metavar='N', type=int, default=3, help='Remove sentences with less than N tokens (defaults to 3)')
    preprocessing_group.add_argument('--max-tokens', metavar='N', type=int, default=80, help='Remove sentences with more than N tokens (defaults to 80)')
    preprocessing_group.add_argument('--dev-size', metavar='N', type=int, default=2000, help='Number of sentences for tuning (defaults to 2000)')

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

    vecmap_group = parser.add_argument_group('Step 4', 'Embedding mapping')
    vecmap_group.add_argument('--vecmap-mode', choices=['identical', 'unsupervised'], default='identical', help='VecMap mode (defaults to identical)')

    induce_group = parser.add_argument_group('Step 5', 'Phrase-table induction')
    induce_group.add_argument('--no-levenshtein', action='store_true', help='Do not include Levenshtein distance in the initial phrase-table')

    tuning_group = parser.add_argument_group('Step 7', 'Unsupervised tuning')
    tuning_group.add_argument('--tuning-iter', metavar='N', type=int, default=10, help='Number of unsupervised tuning iterations (defaults to 10)')
    tuning_group.add_argument('--supervised-tuning', metavar='PATH', nargs=2, help='Parallel corpus for supervised tuning (source/target)')  # TODO Also used for iterative backtranslation
    tuning_group.add_argument('--length-init', action='store_true', help='use length-based initialization')

    backtranslation_group = parser.add_argument_group('Step 8', 'Iterative backtranslation')
    backtranslation_group.add_argument('--backtranslation-iter', metavar='N', type=int, default=3, help='Number of backtranslation iterations (defaults to 3)')
    backtranslation_group.add_argument('--backtranslation-sentences', metavar='N', type=int, default=10000000, help='Number of sentences for training backtranslation (defaults to 10000000)')
    backtranslation_group.add_argument('--no-backtranslation-tuning', action='store_true', help='Disable unsupervised tuning for iterative refinement (use default weights)')

    bitext_group = parser.add_argument_group('Step 9', 'Synthetic parallel corpus generation')
    bitext_group.add_argument('--bpe-tokens', metavar='N', type=int, default=32000, help='BPE vocabulary size')
    bitext_group.add_argument('--bitext-sentences', metavar='N', type=int, default=15000000, help='Number of sentences for bitext generation (defaults to 15000000)')

    nmt_group = parser.add_argument_group('Step 10', 'NMT training')
    nmt_group.add_argument('--nmt-iter', metavar='N', type=int, default=60, help='Number of NMT training iterations (defaults to 60)')
    nmt_group.add_argument('--nmt-sentences-per-iter', metavar='N', type=int, default=1000000, help='Number of sentences for each NMT training iteration (defaults to 1000000)')
    nmt_group.add_argument('--nmt-save-interval', metavar='N', type=int, default=1, help='Save a checkpoint every N iterations (defaults to 1)')
    nmt_group.add_argument('--nmt-transition-iter', metavar='N', type=int, default=30, help='Number of transition iterations between SMT and NMT backtranslation (defaults to 30)')
    nmt_group.add_argument('--nmt-cumul', metavar='N', type=int, default=2, help='Cumulate gradients over N backwards (defaults to 2)')
    nmt_group.add_argument('--nmt-gpus', nargs='+', metavar='N', type=int, default=[0, 1, 2, 3], help='GPU IDs for NMT training (defaults to 0 1 2 3)')
    nmt_group.add_argument('--nmt-fp16', action='store_true', help='Enable FP16 training')

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
            tuning(args)
        if args.from_step <= 8 <= args.to_step:
            iterative_backtranslation(args)
        if args.from_step <= 9 <= args.to_step:
            generate_bitext(args)
        if args.from_step <= 10 <= args.to_step:
            train_nmt(args)

if __name__ == '__main__':
    main()
