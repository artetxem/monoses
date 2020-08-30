import argparse
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract a dictionary from a phrase table')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    parser.add_argument('-f', '--feature', default=2, type=int, help='the index of the feature to use for scoring (defaults to 2)')
    parser.add_argument('-r', '--reverse', action='store_true', help='reverse the dictionary direction (trg -> src)')
    parser.add_argument('-p', '--phrases', action='store_true', help='include phrases in the dictionary')
    args = parser.parse_args()

    fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    for line in fin:
        cols = line.split('|||')
        src, trg = cols[0].strip(), cols[1].strip()
        if args.reverse:
            src, trg = trg, src
        if (' ' not in src and ' ' not in trg) or args.phrases:
            print(src, trg, cols[2].strip().split()[args.feature], sep='\t', file=fout)
    fin.close()
    fout.close()


if __name__ == '__main__':
    main()
