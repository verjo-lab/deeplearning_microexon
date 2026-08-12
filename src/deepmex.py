"""DeepMEx: predict human microexons from a genomic coordinate.

The module is also importable, so other tools (see ``positionscore.py``) can
reuse ``Microexon`` without triggering the command line interface.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pyBigWig
import pybedtools

FLANK_SIZE = 100

base_encode = {
    'A': [0, 0, 0, 1],
    'C': [0, 0, 1, 0],
    'T': [0, 1, 0, 0],
    'G': [1, 0, 0, 0],
    'N': [0, 0, 0, 0]
}


GENOME = None
CONSERVATION_FILE = None


def set_reference_files(genome, conservation):
    global GENOME, CONSERVATION_FILE
    GENOME = genome
    CONSERVATION_FILE = conservation


def load_cnn_model(model_file):
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from keras.models import load_model
    return load_model(model_file)


def capture_values_bw(big_wig, chr, start, end):
    try:
        with pyBigWig.open(big_wig) as bw:
            values_vector  = bw.values(chr, start, end)
            return values_vector
    except Exception as e:
        print(chr,start,end)
        print(e)

def capture_fasta_sequences(fasta_file, chr, start, end):
    fasta_query = pybedtools.BedTool("{} {} {}".format(chr, start, end), from_string=True)
    fasta_query_out = fasta_query.seq((chr, start, end), fasta_file)
    return fasta_query_out

class Microexon():
    def __init__(self, chr, start, end, strand):
        if strand not in ('+', '-'):
            raise ValueError("strand must be '+' or '-', got {!r}".format(strand))
        self.chr  = chr
        self.start  = start
        self.end  = end
        self.strand  = strand
        self.flank_up_limit_start = self.start - 1
        self.flank_up_limit_end = self.start - 101
        self.flank_down_limit_start = self.end + 1
        self.flank_down_limit_end = self.end + 101
        self.conservation_values_up = None
        self.conservation_values_down = None
        self.seq_values_up = None
        self.seq_values_down = None
        self.set_values()
        self.encode_dna_up = self.encoding_dna_sequence(self.seq_values_up)
        self.encode_dna_down = self.encoding_dna_sequence(self.seq_values_down)

    def set_values(self):
        self.conservation_values_up = np.array(
            capture_values_bw(CONSERVATION_FILE,
                self.chr,
                self.flank_up_limit_end,
                self.flank_up_limit_start
            )
        ).reshape(1,100,1)

        self.conservation_values_down = np.array(
            capture_values_bw(CONSERVATION_FILE,
                self.chr,
                self.flank_down_limit_start,
                self.flank_down_limit_end
            )
        ).reshape(1,100,1)

        self.seq_values_up = capture_fasta_sequences(
            GENOME,
            self.chr,
            self.flank_up_limit_end,
            self.flank_up_limit_start
        )

        self.seq_values_down = capture_fasta_sequences(
            GENOME,
            self.chr,
            self.flank_down_limit_start,
            self.flank_down_limit_end
        )

    def encoding_dna_sequence(self, x):
        return np.array([base_encode[x_nucleo.upper()] for x_nucleo in x ])

    def predict_batch(self, model, encode_dna_up, encode_dna_down, verbose=0):
        '''Score a batch of sequence variants for this microexon.

        ``encode_dna_up``/``encode_dna_down`` are (n, 100, 4) arrays; the
        conservation vectors of the microexon are broadcast to the batch.
        Strand handling reproduces the original single-exon code: on the minus
        strand the flanks are swapped and the sequences are read backwards.
        '''
        encode_dna_up = np.asarray(encode_dna_up).reshape(-1, 100, 4)
        encode_dna_down = np.asarray(encode_dna_down).reshape(-1, 100, 4)
        n = encode_dna_up.shape[0]
        conservation_up = np.repeat(self.conservation_values_up.reshape(1, 100, 1), n, axis=0)
        conservation_down = np.repeat(self.conservation_values_down.reshape(1, 100, 1), n, axis=0)

        if self.strand == '+':
            inputs = [encode_dna_up, conservation_up, encode_dna_down, conservation_down]
        else:
            inputs = [encode_dna_down[:, ::-1], conservation_down,
                      encode_dna_up[:, ::-1], conservation_up]

        return model.predict(inputs, verbose=verbose)[:, 0] * 100

    def prediction(self, model):
        '''I need to flip strands at this point'''
        return float(self.predict_batch(model, self.encode_dna_up, self.encode_dna_down)[0])


def terminal_width(default=80):
    try:
        return int(os.popen('stty size', 'r').read().split()[1])
    except Exception:
        return default


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="CNN model file")
    parser.add_argument("--genome", help="HG38 genome file")
    parser.add_argument("--conservation", help="conservation bigwig file hg38.100way.phastCons.bw")
    parser.add_argument("--exon", help="A exon coordinate Ex: chr1:100020:100030:+")

    args = parser.parse_args(argv)

    set_reference_files(args.genome, args.conservation)
    model_merged = load_cnn_model(args.model)

    coord = args.exon.split(':')
    print(coord)
    m1 = Microexon(coord[0], int(coord[1]), int(coord[2]), coord[3])
    prediction = m1.prediction(model_merged)
    if sys.stdout.isatty():
        os.system('clear' if os.name == 'posix' else 'cls')
    print('=' * terminal_width())
    print('Microexon prediction: {}\n\n\n'.format(prediction))


if __name__ == '__main__':
    main()
