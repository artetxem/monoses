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
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter a tokenized corpus by the number of tokens per line')
    parser.add_argument('--min', type=int, default=1, help='minimum tokens per line')
    parser.add_argument('--max', type=int, default=80, help='maximum tokens per line')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    args = parser.parse_args()

    fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    for line in fin:
        n = len(line.strip().split())
        if args.min <= n <= args.max:
            print(line, end='', file=fout)
    fin.close()
    fout.close()


if __name__ == '__main__':
    main()
