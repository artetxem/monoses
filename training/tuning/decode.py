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
import subprocess
import sys
import tempfile
from shlex import quote

import tune


def bash(command):
    subprocess.run(['bash', '-c', command])


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate an n-best list for tuning')
    parser.add_argument('-o', '--output', required=True, help='the output n-best list')

    parser.add_argument('--bt2ref', nargs=2, required=True, help='backtranslated corpus in the source language and reference in the target language')
    parser.add_argument('--src', required=False)

    parser.add_argument('--cache', required=True)

    parser.add_argument('--config', required=True, help='the src2trg moses configuration file that is being tuned')
    parser.add_argument('--weights', required=True, help='the zmert weights file')
    parser.add_argument('--config-bwd', required=False, help='the trg2src moses configuration file that is fixed during tuning (optional, used for backward -expensive- BLEU)')
    parser.add_argument('--moses', required=True, help='path to moses')
    parser.add_argument('--nbest', type=int, default=100, help='size of the n-best list')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='number of threads (defaults to 20)')
    parser.add_argument('--cube-pruning-pop-limit', metavar='N', type=int, default=1000, help='Cube pruning pop limit for backward decoding (defaults to 1000)')
    parser.add_argument('--lm-feature', default='LM0', help='the language model feature name')
    args = parser.parse_args()

    cache = {}
    with open(args.cache, encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            src, trg = line.strip().split('\t')
            cache[src] = trg

    with tempfile.TemporaryDirectory() as tmp:
        zmert_params = tune.extract_zmert_params(args.weights)
        tune.replace_moses_params(args.config, tmp + '/moses.ini', zmert_params)

        subprocess.run([args.moses + '/bin/moses2',
                        '-i', args.bt2ref[0],
                        '-f', tmp + '/moses.ini',
                        '--n-best-list', tmp + '/bt2sys.txt', str(args.nbest), 'distinct',
                        '--threads', str(args.threads)])

        if args.src is not None:
            subprocess.run([args.moses + '/bin/moses2',
                            '-i', args.src,
                            '-f', tmp + '/moses.ini',
                            '--n-best-list', tmp + '/src2sys.txt', str(args.nbest), 'distinct',
                            '--threads', str(args.threads)])

            # src2sys -> src
            sentences = set()
            with open(tmp + '/src2sys.txt', encoding='utf-8', errors='surrogateescape') as f:
                for line in f:
                    translation = line.strip().split(' ||| ')[1].strip()
                    if translation not in cache:
                        sentences.add(translation)
            with open(tmp + '/input.txt', mode='w', encoding='utf-8', errors='surrogateescape') as f:
                for sentence in sentences:
                    print(sentence, file=f)
            bash(quote(args.moses + '/bin/moses2') + ' -i ' + quote(tmp + '/input.txt') +
                 ' -f ' + quote(args.config_bwd) +
                 ' -search-algorithm 1 -cube-pruning-pop-limit {0} -s {0}'.format(args.cube_pruning_pop_limit) +
                 ' --threads ' + str(args.threads) +
                 ' > ' + quote(tmp + '/output.txt'))
            with open(tmp + '/input.txt', encoding='utf-8', errors='surrogateescape') as fsrc:
                with open(tmp + '/output.txt', encoding='utf-8', errors='surrogateescape') as ftrg:
                    with open(args.cache, mode='a', encoding='utf-8', errors='surrogateescape') as fcache:
                        for src, trg in zip(fsrc, ftrg):
                            src, trg = src.strip(), trg.strip()
                            cache[src] = trg
                            print(src, trg, sep='\t', file=fcache)

        offset = 0
        with open(args.output, mode='w', encoding='utf-8', errors='surrogateescape') as fout:
            directions = [(tmp + '/bt2sys.txt', True)]
            if args.src is not None:
                directions.append((tmp + '/src2sys.txt', False))
            for filename, bt2ref in directions:
                with open(filename, encoding='utf-8', errors='surrogateescape') as fin:
                    for line in fin:
                        cols = line.strip().split(' ||| ')
                        ind, candidate, weights = int(cols[0].strip()), cols[1].strip(), cols[2].strip()
                        if bt2ref:
                            offset = ind + 1
                        else:
                            ind += offset

                        params = collections.defaultdict(list)
                        name = None
                        for tok in weights.split():
                            tok = tok.strip()
                            if tok[-1] == '=':
                                name = tok[:-1]
                            else:
                                params[name].append(tok)
                        for name in zmert_params.keys():
                            if name not in params:
                                params[name] = ['0' for i in range(len(zmert_params[name]))]

                        weights = ' '.join([weight for name in sorted(params.keys()) for weight in params[name]])
                        output = ['-', # weights.replace(' ', '_'),  # Unique score ID
                                  '0' if bt2ref else '1',  # Direction ID
                                  candidate if bt2ref else cache[candidate],  # Translation
                                  '-' if bt2ref else params[args.lm_feature][0],  # Language model score
                                  '-' if bt2ref else candidate  # Language model text
                                  ]
                        print(' ||| '.join([str(ind), '\t'.join(output), weights]), file=fout)


if __name__ == '__main__':
    main()
