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

import train


ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.abspath(os.environ['MONOSES_THIRD_PARTY']) if 'MONOSES_THIRD_PARTY' in os.environ else ROOT + '/third-party'
MOSES = THIRD_PARTY + '/moses'


def bash(command):
    subprocess.run(['bash', '-c', command])


def main():
    parser = argparse.ArgumentParser(description='Translate text using a trained model')
    parser.add_argument('model', metavar='PATH', help='Working directory of the trained model')
    parser.add_argument('-r', '--reverse', action='store_true', help='Use the reverse model (trg->src)')
    parser.add_argument('--src', metavar='STR', required=True, help='Input language code')
    parser.add_argument('--trg', metavar='STR', required=True, help='Output language code')
    parser.add_argument('--tok', action='store_true', help='Tokenized input/output')
    parser.add_argument('--step', metavar='N', type=int, default=10, help='Step number (defaults to 10)')
    parser.add_argument('--nmt-checkpoints', nargs='+', metavar='N', default=[10, 20, 30, 40, 50, 60], help='Use a checkpoint ensemble over the given iterations')
    parser.add_argument('--threads', metavar='N', type=int, default=20, help='Number of threads (defaults to 20)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU decoding')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 decoding')
    args = parser.parse_args()

    direction = 'trg2src' if args.reverse else 'src2trg'

    command = 'cat -'
    if not args.tok:
        command += ' | ' + train.tokenize_command(args, args.src)
    command += ' | ' + quote(MOSES + '/scripts/recaser/truecase.perl')
    command += ' --model ' + quote(args.model + '/step1/truecase-model.' + direction[:3])
    if args.step == 10:
        command += ' | python3 ' + quote(train.SUBWORD_NMT + '/subword_nmt/apply_bpe.py') + ' -c ' + quote(args.model + '/step9/bpe.codes')
        command += ' | python3 ' + quote(train.FAIRSEQ + '/interactive.py') + ' ' + quote(args.model + '/step10/data.bin')
        command += ' --path '
        command += ':'.join([quote(args.model + '/step10/' + direction + '.' + str(it) + '.pt') for it in args.nmt_checkpoints])
        command += ' --source-lang src --target-lang trg'
        command += ' --beam 5'
        command += ' --max-tokens 1000'
        command += ' --buffer-size 10000'
        if args.cpu:
            command += ' --cpu'
        if args.fp16:
            command += ' --fp16'
        command += ' | grep -P \'^H\t\''
        command += ' | cut -f3'
        command += ' | sed -r \'s/(@@ )|(@@ ?$)//g\''
    else:
        command += ' | ' + quote(MOSES + '/bin/moses2')
        if args.step == 6:
            command += ' -f ' + quote(args.model + '/step6/' + direction + '.moses.ini')
        elif args.step == 7:
            command += ' -f ' + quote(args.model + '/step7/' + direction + '.moses.ini')
        elif args.step == 8:
            command += ' -f ' + quote(args.model + '/step8/' + direction + '.moses.ini')
        command += ' --threads ' + str(args.threads)
        command += ' 2> /dev/null'
    command += ' | ' + quote(MOSES + '/scripts/recaser/detruecase.perl')
    if not args.tok:
        command += ' | ' + quote(MOSES + '/scripts/tokenizer/detokenizer.perl') + ' -q -l ' + quote(args.trg)

    bash(command)


if __name__ == '__main__':
    main()
