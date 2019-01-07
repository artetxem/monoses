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
import collections
import filecmp
import math
import os
import shutil
import subprocess
import sys
import tempfile
from shlex import quote

ROOT = os.path.dirname(os.path.abspath(__file__))
ZMERT = ROOT + '/zmert/build/tune.jar'

BLEU_THRESHOLD = 3.0
RATIO_THRESHOLD = 0.002
PENALTY_THRESHOLD = 0.01
PENALTY_MIN = -3.5
PENALTY_MAX = 1.0
PENALTY_DELTA = 0.1


def bash(command):
    subprocess.run(['bash', '-c', command])


def extract_feature_path(config_path, feature_name):
    start = False
    with open(config_path, encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            line = line.strip()
            if line == '[feature]':
                start = True
            elif start and line != '':
                cols = line.split()
                fields = {col.split('=')[0]: col.split('=')[1] for col in cols if '=' in col}
                if 'name' in fields and 'path' in fields and fields['name'] == feature_name:
                    return fields['path']
    return None


def extract_moses_params(path):
    params = {}
    start = False
    with open(path, encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            line = line.strip()
            if line == '[weight]':
                start = True
            elif start and line != '':
                cols = line.split()
                params[cols[0][:-1]] = [param for param in cols[1:]]
    return params


def extract_zmert_params(path):
    name2ind2weight = collections.defaultdict(dict)
    with open(path, encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            line = line.strip()
            if line != '':
                cols = line.split()
                name = cols[0].split('___')[0]
                ind  = cols[0].split('___')[1]
                name2ind2weight[name][int(ind)] = cols[1]
    params = collections.defaultdict(list)
    for name in name2ind2weight.keys():
        ind = 0
        while True:
            if ind in name2ind2weight[name]:
                params[name].append(name2ind2weight[name][ind])
                ind += 1
            else:
                break
    return params


def replace_moses_params(input, output, params):
    start = False
    with open(input, encoding='utf-8', errors='surrogateescape') as fin:
        with open(output, mode='x', encoding='utf-8', errors='surrogateescape') as fout:
            for line in fin:
                line = line.strip()
                if start and line != '':
                    name = line.split()[0][:-1]
                    print(name + '= ' + ' '.join([weight for weight in params[name]]), file=fout)
                else:
                    print(line, file=fout)
                if line == '[weight]':
                    start = True


def translate_command(args, config, word_penalty=None, cube_pruning_pop_limit=None):
    ans = quote(os.path.abspath(args.moses + '/bin/moses2'))
    ans += ' -f ' + os.path.abspath(config)
    ans += ' --threads ' + str(args.threads)
    if cube_pruning_pop_limit is not None:
        ans += ' -search-algorithm 1 -cube-pruning-pop-limit {0} -s {0}'.format(cube_pruning_pop_limit)
    if word_penalty is not None:
        ans += ' --weight-overwrite \'{}= {}\''.format(args.word_penalty_feature, word_penalty)
    ans += ' 2> /dev/null'
    return ans


def word_count(f):
    return int(subprocess.run(['bash', '-c', 'cat ' + quote(f) + ' | wc -w'],
                              stdout=subprocess.PIPE).stdout.decode('utf-8').strip())


def multi_bleu(args, sys, ref):
    cols = subprocess.run(['bash', '-c', 'cat ' + quote(sys) +
                           ' | ' + quote(args.moses + '/scripts/generic/multi-bleu.perl') + ' ' + quote(ref)],
                          stdout=subprocess.PIPE).stdout.decode('utf-8').strip().split(' ')
    return {'bleu': float(cols[2][:-1]), 'ratio': float(cols[5][6:-1])}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tune Moses weights')
    parser.add_argument('--dev', nargs=2, required=True, help='the development corpus (src & trg)')
    parser.add_argument('--input', nargs=2, required=True, help='the input configuration files (src2trg & trg2src)')
    parser.add_argument('--output', nargs=2, required=True, help='the output configuration files (src2trg & trg2src)')
    parser.add_argument('--moses', required=True, help='path to moses')
    parser.add_argument('--supervised', action='store_true', help='standard supervised tuning over parallel corpora')
    parser.add_argument('--nbest', type=int, default=100, help='size of the n-best list (defaults to 100)')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='number of threads (defaults to 20)')
    parser.add_argument('--iterations', metavar='N', type=int, default=10, help='perform N iterations for unsupervised tuning (defaults to 10)')
    parser.add_argument('--cube-pruning-pop-limit', metavar='N', type=int, default=1000, help='cube pruning pop limit for fast decoding (defaults to 1000)')
    parser.add_argument('--word-penalty-feature', default='WordPenalty0', help='the word penalty feature name')
    parser.add_argument('--lm-feature', default='LM0', help='the language model feature name')
    parser.add_argument('--length-init', action='store_true', help='use length-based initialization')
    # TODO Add option to specify tmp dir
    args = parser.parse_args()

    if args.supervised or not args.length_init:
        fwdind, bwdind = 0, 1
        shutil.copy(args.input[0], args.output[0] + '.it0')
        shutil.copy(args.input[1], args.output[1] + '.it0')
    else:
        print('Identifying shortest language...', file=sys.stderr)
        ratios = [None, None]
        for i in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                bash('cat ' + quote(args.dev[i]) +
                     ' | ' + translate_command(args, args.input[i], cube_pruning_pop_limit=args.cube_pruning_pop_limit) +
                     ' > ' + quote(tmp + '/output.txt'))
                ratios[i] = word_count(args.dev[i]) / word_count(tmp + '/output.txt')
        print('  * src2trg ratio: {:.4f}'.format(ratios[0]), file=sys.stderr)
        print('  * trg2src ratio: {:.4f}'.format(ratios[1]), file=sys.stderr)
        if ratios[0] > ratios[1]:
            fwdind, bwdind = 1, 0
            print('trg seems to be the shortest language!', file=sys.stderr)
        else:
            fwdind, bwdind = 0, 1
            print('src seems to be the shortest language!', file=sys.stderr)
        print(file=sys.stderr)

        print('Optimizing word penalties...', file=sys.stderr)
        best_penalties = [None, None]
        best_bleu = -1.0
        fwdpenalty = bwdpenalty = PENALTY_MIN
        while fwdpenalty <= PENALTY_MAX + PENALTY_THRESHOLD:
            pmin = pmax = None
            pinitial = bwdpenalty
            delta = PENALTY_DELTA
            with tempfile.TemporaryDirectory() as tmp:
                bash('cat ' + quote(args.dev[fwdind]) +
                     ' | ' + translate_command(args, args.input[fwdind], word_penalty=fwdpenalty,
                                               cube_pruning_pop_limit=args.cube_pruning_pop_limit) +
                     ' > ' + quote(tmp + '/dev.bt'))
                while True:
                    bash('cat ' + quote(tmp + '/dev.bt') +
                         ' | ' + translate_command(args, args.input[bwdind], word_penalty=bwdpenalty,
                                                   cube_pruning_pop_limit=args.cube_pruning_pop_limit) +
                         ' > ' + quote(tmp + '/dev.sys'))
                    stats = multi_bleu(args, sys=tmp+'/dev.sys', ref=args.dev[fwdind])
                    ratio = stats['ratio']
                    if abs(1.0 - ratio) < RATIO_THRESHOLD:
                        break
                    if ratio > 1.0:
                        pmin = bwdpenalty
                    else:
                        pmax = bwdpenalty
                    if pmin is None:
                        bwdpenalty = pinitial - delta
                        delta *= 2
                    elif pmax is None:
                        bwdpenalty = pinitial + delta
                        delta *= 2
                    else:
                        bwdpenalty = pmin + (pmax - pmin) / 2
                        if pmax - pmin < PENALTY_THRESHOLD:
                            break
            print('  * {:.2f}/{:.2f}: {:.2f} BLEU'.format(fwdpenalty, bwdpenalty, stats['bleu']), file=sys.stderr)
            if stats['bleu'] >= best_bleu:
                best_bleu = stats['bleu']
                best_penalties[fwdind] = fwdpenalty
                best_penalties[bwdind] = bwdpenalty
            elif stats['bleu'] < best_bleu - BLEU_THRESHOLD:
                break
            fwdpenalty += PENALTY_DELTA
            bwdpenalty -= PENALTY_DELTA
        print('best penalties: {:.2f}/{:.2f} ({:.2f} BLEU)'.format(best_penalties[0], best_penalties[1], best_bleu), file=sys.stderr)
        print(file=sys.stderr)
        for i in range(2):
            params = extract_moses_params(args.input[i])
            params[args.word_penalty_feature] = ['{:.4f}'.format(best_penalties[i])]
            replace_moses_params(args.input[i], args.output[i] + '.it0', params)

    print('Estimating LM entropies in dev...', file=sys.stderr)
    dev_entropies = [0.0, 0.0]
    for i, part in enumerate(['src', 'trg']):
        output = subprocess.run(['bash', '-c', quote(args.moses + '/bin/query') + ' -v summary ' +
                                 quote(extract_feature_path(args.input[(i+1)%2], args.lm_feature)) +
                                 ' < ' + quote(args.dev[i])],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')
        dev_entropies[i] = math.log(float(output.splitlines()[0].split()[-1]))
        print('  * {} entropy: {:.4f}'.format(part, dev_entropies[i]), file=sys.stderr)
    print(file=sys.stderr)

    early_stop = False
    iterations = 1 if args.supervised else args.iterations
    for it in range(iterations):
        if early_stop:
            break
        for is_first, ind, rind in (True, fwdind, bwdind), (False, bwdind, fwdind):
            if early_stop:
                break
            srcdev = args.dev[ind]
            trgdev = args.dev[rind]
            trgentropy = dev_entropies[rind]
            src2trg_config = args.output[ind] + '.it' + str(it)
            trg2src_config = args.output[rind] + '.it' + str(it + (0 if is_first else 1))

            with tempfile.TemporaryDirectory() as tmp:
                open(tmp + '/cache.txt', mode='x').close()

                # Build reference
                if args.supervised:
                    shutil.copy(srcdev, tmp + '/input.txt')
                    shutil.copy(trgdev, tmp + '/reference.txt')
                else:
                    bash('cat ' + quote(trgdev) +
                         ' | ' + translate_command(args, trg2src_config) +
                         ' > ' + quote(tmp + '/input.txt'))
                    bash('cat ' + quote(trgdev) + ' ' + quote(srcdev) + ' > ' + quote(tmp + '/reference.txt'))

                # Build params file
                params = extract_moses_params(src2trg_config)
                with open(quote(tmp + '/params.txt'), mode='w', encoding='utf-8', errors='surrogateescape') as f:
                    for name in sorted(params.keys()):
                        for i, val in enumerate(params[name]):
                            print('{}___{} ||| {} Opt -Inf +Inf -1 +1'.format(name, i, val), file=f)
                    print('normalization = LNorm 1 1', file=f)

                # Build decoder configuration file
                with open(quote(tmp + '/dcfg.txt'), mode='w', encoding='utf-8', errors='surrogateescape') as f:
                    for name in sorted(params.keys()):
                        for i, val in enumerate(params[name]):
                            print('{}___{} ||| {}'.format(name, i, val), file=f)

                # Build decoder launcher
                with open(quote(tmp + '/cmd.sh'), mode='w', encoding='utf-8', errors='surrogateescape') as f:
                    print('#!/bin/bash', file=f)
                    print('python3 ' + quote(ROOT + '/decode.py') +
                          ' --bt2ref ' + quote(tmp + '/input.txt') + ' ' + quote(trgdev) +
                          ('' if args.supervised else (' --src ' + quote(srcdev)))  +
                          ' -o ' + quote(tmp + '/output.txt') +
                          ' --cache ' + quote(tmp + '/cache.txt') +
                          ' --config ' + quote(src2trg_config) +
                          ' --config-bwd ' + quote(trg2src_config) +
                          ' --weights ' + quote(tmp + '/dcfg.txt') +
                          ' --moses ' + quote(args.moses) +
                          ' --lm-feature ' + quote(args.lm_feature) +
                          ' --threads ' + str(args.threads) +
                          ' --cube-pruning-pop-limit ' + str(args.cube_pruning_pop_limit) +
                          ' --nbest ' + str(args.nbest-1),  # For some reason moses returns one extra entry
                          file=f)
                    os.chmod(f.fileno(), 0o700)

                # Build ZMERT configuration file
                config_path = tmp + '/config.txt'
                with open(config_path, mode='w', encoding='utf-8', errors='surrogateescape') as f:
                    print('-dir ' + quote(tmp), file=f)
                    print('-r reference.txt', file=f)
                    print('-p params.txt', file=f)
                    print('-cmd cmd.sh', file=f)
                    print('-decOut output.txt', file=f)
                    print('-dcfg dcfg.txt', file=f)
                    print('-txtNrm 0', file=f)  # Do not normalize (tokenize) text
                    print('-rps 1', file=f)  # References per sentence
                    print('-m monoses 4 closest {}'.format(trgentropy), file=f)
                    print('-maxIt 8', file=f)  # TODO Parameterize this
                    print('-ipi 20', file=f)  # TODO Parameterize this  # Number of intermediate initial points per iteration
                    print('-N {}'.format(args.nbest), file=f)  # Size of N-best list generated each iteration
                    print('-v 1', file=f)  # Verbosity level (0-2)
                    print('-seed 1', file=f)  # TODO Parameterize this  # Random seed
                    print('-thrCnt {}'.format(args.threads), file=f)
                    # print('-decV 1', file=f)  # Print decoder output

                subprocess.run(['java', '-jar', ZMERT, '-maxMem', '16384', config_path])  # TODO Adjust memory

                replace_moses_params(args.output[ind] + '.it' + str(it),
                                     args.output[ind] + '.it' + str(it+1),
                                     extract_zmert_params(tmp + '/dcfg.txt.ZMERT.final'))

                if filecmp.cmp(args.output[ind] + '.it' + str(it), args.output[ind] + '.it' + str(it+1), shallow=False):
                    shutil.copy(src2trg_config, args.output[ind])
                    shutil.copy(trg2src_config, args.output[rind])
                    early_stop = True

    if not early_stop:
        shutil.copy(args.output[0] + '.it' + str(iterations), args.output[0])
        shutil.copy(args.output[1] + '.it' + str(iterations), args.output[1])


if __name__ == '__main__':
    main()
