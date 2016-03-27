# -*- coding:utf-8 -*- 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np

indices = ['v23', 'v10', 'v50', 'v114', 'v12', 'v14', 'v21', 'v34', 'v40']

### Constants
IS_WINDOWS = True if system() == 'Windows' else False
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'windows', 'bh_tsne.exe') if IS_WINDOWS else path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
    'same directory as this script, have you forgotten to compile it?: {}'
    ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2014)
# https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf (Experimental Setup, page 13)
DEFAULT_NO_DIMS = 2
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 50
DEFAULT_THETA = 0.5
EMPTY_SEED = -1

###

def _argparse():
    argparse = ArgumentParser('bh_tsne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int,
                          default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float,
            default=DEFAULT_PERPLEXITY)
    # 0.0 for theta is equivalent to vanilla t-SNE
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)
    argparse.add_argument('-r', '--randseed', type=int, default=EMPTY_SEED)
    argparse.add_argument('-n', '--initial_dims', type=int, default=INITIAL_DIMENSIONS)
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'),
            default=stdout)
    return argparse


class TmpDir:
    def __enter__(self):
        self._tmp_dir_path = mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        rmtree(self._tmp_dir_path)


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

def bh_tsne(samples, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS, perplexity=DEFAULT_PERPLEXITY,
            theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False):

    samples -= np.mean(samples, axis=0)
    cov_x = np.dot(np.transpose(samples), samples)
    [eig_val, eig_vec] = np.linalg.eig(cov_x)

    # sorting the eigen-values in the descending order
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

    if initial_dims > len(eig_vec):
        initial_dims = len(eig_vec)

    # truncating the eigen-vectors matrix to keep the most important vectors
    eig_vec = eig_vec[:, :initial_dims]
    samples = np.dot(samples, eig_vec)

    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    with TmpDir() as tmp_dir_path:
        # Note: The binary format used by bh_tsne is roughly the same as for
        #   vanilla tsne
        with open(path_join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
            # Then write the data
            for sample in samples:
                data_file.write(pack('{}d'.format(len(sample)), *sample))
            # Write random seed if specified
            if randseed != EMPTY_SEED:
                data_file.write(pack('i', randseed))

        # Call bh_tsne and let it do its thing
        with open(devnull, 'w') as dev_null:
            bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH), ), cwd=tmp_dir_path,
                    # bh_tsne is very noisy on stdout, tell it to use stderr
                    #   if it is to print any output
                    stdout=stderr if verbose else dev_null)
            bh_tsne_p.wait()
            assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                    'with a non-zero return code exit status, please ' +
                    ('enable verbose mode and ' if not verbose else '') +
                    'refer to the bh_tsne output for further details')

        # Read and pass on the results
        with open(path_join(tmp_dir_path, 'result.dat'), 'rb') as output_file:
            # The first two integers are just the number of samples and the
            #   dimensionality
            result_samples, result_dims = _read_unpack('ii', output_file)
            # Collect the results, but they may be out of order
            results = [_read_unpack('{}d'.format(result_dims), output_file)
                for _ in xrange(result_samples)]
            # Now collect the landmark data so that we can return the data in
            #   the order it arrived
            results = [(_read_unpack('i', output_file), e) for e in results]
            # Put the results in order and yield it
            results.sort()
            for _, result in results:
                yield result
            # The last piece of data is the cost for each sample, we ignore it
            #read_unpack('{}d'.format(sample_count), output_file)

def main(filename):
    data = pd.read_csv(filename, index_col=0)
    X = np.array(data[indices]).tolist()
    y = np.array(data['target'])
    # tsne = TSNE(n_components=2)
    # reduced = tsne.fit_transform(X)
    # np.savetxt(filename+'.tsne2d.txt', reduced)
    # plt.scatter(reduced[:, 0], reduced[:, 1], cmap=y)
    # plt.savefig('tsne.png')

    result = list(bh_tsne(X, no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=True, initial_dims=len(indices)))
    np.savetxt(filename+'.tsne2d.txt', result)
    result = np.array(result)
    plt.scatter(result[:, 0], result[:, 1], c=['r', 'b'], cmap=y)
    plt.savefig(filename+'.tsne.png')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main('../train.t1.csv')
