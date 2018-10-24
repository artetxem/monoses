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
import collections
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract ngrams from a text file')
    parser.add_argument('--min-order', type=int, default=1)
    parser.add_argument('--max-order', type=int, default=3)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    args = parser.parse_args()

    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    ngram2count = collections.defaultdict(int)
    for line in f:
        tokens = line.strip().split()
        n = len(tokens)
        for i in range(n):
            for j in range(i + args.min_order, min(i + args.max_order + 1, n + 1)):
                ngram2count[' '.join(tokens[i:j])] += 1
    f.close()

    f = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    for ngram, count in ngram2count.items():
        if count >= args.min_count:
            print('{0}\t{1}'.format(count, ngram), file=f)
    f.close()


if __name__ == '__main__':
    main()
