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
import numpy as np
import sys
import torch
import torch.nn.functional as F


def read_embeddings(file, dtype='float32'):
    header = file.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return (words, matrix)


def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]


def compute_loss(x, z, src, trg, temperature):
    return F.cross_entropy(x[src].matmul(z.t())/temperature, trg, size_average=True)


def unigram_dictionary(x, z, words_x, words_z, temperature, batch_size, min_prob, device):
    unigrams_x = torch.tensor([i for i, word in enumerate(words_x) if not '&#32;' in word]).to(device)
    unigrams_z = torch.tensor([i for i, word in enumerate(words_z) if not '&#32;' in word]).to(device)
    src2trg2prob = collections.defaultdict(lambda: collections.defaultdict(lambda: min_prob))
    for i in range(0, unigrams_x.shape[0], batch_size):
        j = min(i + batch_size, unigrams_x.shape[0])
        probs, ind = F.softmax(x[unigrams_x[i:j]].matmul(z[unigrams_z].t())/temperature, dim=1).sort(dim=1, descending=True)
        probs = probs.detach().cpu().numpy()
        ind = ind.detach().cpu().numpy()
        for k in range(i, j):
            for l, prob in zip(ind[k-i], probs[k-i]):
                if prob <= min_prob:
                    break
                src2trg2prob[words_x[unigrams_x[k]]][words_z[unigrams_z[l]]] = prob
    return src2trg2prob


def write_phrase_table(x, z, words_x, words_z, temperature, size, batch_size, min_prob, device, f):
    with torch.no_grad():
        src2trg2prob = unigram_dictionary(x, z, words_x, words_z, temperature, batch_size, min_prob, device)
        trg2src2prob = unigram_dictionary(z, x, words_z, words_x, temperature, batch_size, min_prob, device)
        partition_x = torch.zeros(x.shape[0], device=device)
        partition_z = torch.zeros(z.shape[0], device=device)
        for i in range(0, x.shape[0], batch_size):
            j = min(i + batch_size, x.shape[0])
            partition_x[i:j] = (x[i:j].matmul(z.t())/temperature).exp().sum(dim=1)
        for i in range(0, z.shape[0], batch_size):
            j = min(i + batch_size, z.shape[0])
            partition_z[i:j] = (z[i:j].matmul(x.t())/temperature).exp().sum(dim=1)
        for i in range(0, x.shape[0], batch_size):
            j = min(i + batch_size, x.shape[0])
            scores, ind = x[i:j].matmul(z.t()).topk(size, dim=1)  # batch*k
            scores = (scores/temperature).exp()
            probs = scores / partition_x[i:j].unsqueeze(1)
            invprobs = scores / partition_z[ind]
            np_probs = probs.detach().cpu().numpy()
            np_ind = ind.detach().cpu().numpy()
            np_invprobs = invprobs.detach().cpu().numpy()
            for k in range(i, j):
                for prob, invprob, l in zip(np_probs[k-i], np_invprobs[k-i], np_ind[k-i]):
                    lexprob = np.prod([max([src2trg2prob[src][trg] for src in words_x[k].split('&#32;')]) for trg in words_z[l].split('&#32;')])
                    invlexprob = np.prod([max([trg2src2prob[trg][src] for trg in words_z[l].split('&#32;')]) for src in words_x[k].split('&#32;')])
                    print('{0} ||| {1} ||| {2} {3} {4} {5} ||| ||| ||| |||'.format(words_x[k].replace('&#32;', ' '), words_z[l].replace('&#32;', ' '), invprob, invlexprob, prob, lexprob), file=f)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Induce a phrase table from a set of cross-lingual phrase embeddings')
    parser.add_argument('--src', help='the source embeddings')
    parser.add_argument('--trg', help='the target embeddings')
    parser.add_argument('--src2trg', help='the output source-to-target phrase table')
    parser.add_argument('--trg2src', help='the output target-to-source phrase table')
    parser.add_argument('--lr', default=3e-4, type=float, help='the learning rate (defaults to 3e-4')
    parser.add_argument('--epochs', default=1, type=int, help='the number of epochs (defaults to 1)')
    parser.add_argument('--batch', default=200, type=int, help='the batch size (defaults to 200)')
    parser.add_argument('--min-prob', default=0.001, type=float, help='minimum translation probability for unigrams (defaults to 0.001)')
    parser.add_argument('--size', default=100, type=int, help='number of target entries for each source phrase (defaults to 100)')
    parser.add_argument('--dot', action='store_true', help='do not length normalize embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', action='store_true', help='use gpu (cuda)')
    args = parser.parse_args()

    # Read input embeddings
    srcfile = open(args.src, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg, encoding=args.encoding, errors='surrogateescape')
    src_words, x = read_embeddings(srcfile, dtype='float32')
    trg_words, z = read_embeddings(trgfile, dtype='float32')
    if not args.dot:
        length_normalize(x)
        length_normalize(z)

    # Initialize pytorch variables
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    x = torch.tensor(x, requires_grad=False).to(device)
    z = torch.tensor(z, requires_grad=False).to(device)
    t = torch.tensor(1.0, device=device, requires_grad=True)

    # Optimize the temperature
    optimizer = torch.optim.Adam([t], lr=args.lr)
    for epoch in range(args.epochs):
        src_x = torch.randperm(x.shape[0], requires_grad=False).to(device)
        src_z = torch.randperm(z.shape[0], requires_grad=False).to(device)
        n = min(x.shape[0], z.shape[0])  # TODO We are taking a subset of the largest side
        for i in range(0, n, args.batch):
            j = min(i + args.batch, x.shape[0])
            trg_z = x[src_x[i:j]].matmul(z.t()).max(dim=1)[1]
            trg_x = z[src_z[i:j]].matmul(x.t()).max(dim=1)[1]
            loss  = compute_loss(x, z, trg_x, src_z[i:j], t)
            loss += compute_loss(z, x, trg_z, src_x[i:j], t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Optimizing temperature | Progress: {:.2%} | Temperature: {:.2f} | Loss: {:.2f}'
                  .format((epoch + j/n)/args.epochs, t.detach().cpu().numpy(), loss.detach().cpu().numpy()),
                  file=sys.stderr)

    if args.src2trg is not None:
        f = open(args.src2trg, mode='w', encoding=args.encoding, errors='surrogateescape')
        write_phrase_table(x, z, src_words, trg_words, t, args.size, args.batch, args.min_prob, device, f)

    if args.trg2src is not None:
        f = open(args.trg2src, mode='w', encoding=args.encoding, errors='surrogateescape')
        write_phrase_table(z, x, trg_words, src_words, t, args.size, args.batch, args.min_prob, device, f)


if __name__ == '__main__':
    main()
