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
import os
import subprocess
from shlex import quote


ROOT = os.path.dirname(os.path.abspath(__file__))
MOSES = ROOT + '/third-party/moses'


def bash(command):
    subprocess.run(['bash', '-c', command])


def main():
    parser = argparse.ArgumentParser(description='Translate text using a trained model')
    parser.add_argument('model', metavar='PATH', help='Working directory of the trained model')
    parser.add_argument('-r', '--reverse', action='store_true', help='Use the reverse model (trg->src)')
    parser.add_argument('--src', metavar='STR', required=True, help='Input language code')
    parser.add_argument('--trg', metavar='STR', required=True, help='Output language code')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='Number of threads (defaults to 20)')
    parser.add_argument('--tok', action='store_true', help='Do not detokenize')
    args = parser.parse_args()

    direction = 'trg2src' if args.reverse else 'src2trg'
    detok = '' if args.tok else ' | ' + quote(MOSES + '/scripts/tokenizer/detokenizer.perl') + ' -q -l ' + quote(args.trg)
    bash(quote(MOSES + '/scripts/tokenizer/tokenizer.perl') +
               ' -l ' + quote(args.src) + ' -threads ' + str(args.threads) +
               ' 2> /dev/null' +
         ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl') +
               ' --model ' + quote(args.model + '/step1/truecase-model.' + direction[:3]) +
         ' | ' + quote(MOSES + '/bin/moses2') +
               ' -f ' + quote(args.model + '/' + direction + '.moses.ini') +
               ' --threads ' + str(args.threads) +
               ' 2> /dev/null' +
         ' | ' + quote(MOSES + '/scripts/recaser/detruecase.perl') +
         detok
         )


if __name__ == '__main__':
    main()
