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
import gzip
import sys

import editdistance


def similarity(s1, s2):
    dist = editdistance.eval(s1, s2)
    return 1 - dist / max(len(s1), len(s2))


def main():
    parser = argparse.ArgumentParser(description='Add Levenshtein distance to a phrase-table')
    parser.add_argument('--min-sim', default=0.3, type=float, help='the minimum similarity (defaults to 0.3)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    args = parser.parse_args()

    fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')

    for line in fin:
        cols = line.split('|||')
        src = cols[0].strip().split()
        trg = cols[1].strip().split()

        score_bwd = 1.0
        for t1 in src:
            best = args.min_sim
            for t2 in trg:
                sim = similarity(t1, t2)
                best = max(best, sim)
            score_bwd *= best
        cols[2] = cols[2] + str(score_bwd) + ' '

        score_fwd = 1.0
        for t1 in trg:
            best = args.min_sim
            for t2 in src:
                sim = similarity(t1, t2)
                best = max(best, sim)
            score_fwd *= best
        cols[2] = cols[2] + str(score_fwd) + ' '

        print('|||'.join(cols), end='', file=fout)
    
    fin.close()
    fout.close()


if __name__ == '__main__':
    main()
