"""Checks for the mutational screening, with a stand in model.

Runs without tensorflow, without the reference files and without a GPU:
pyBigWig/pybedtools are stubbed and the CNN is replaced by a deterministic
scoring function, so only the wiring is under test.

    python tests/test_positionscore.py
"""

from __future__ import print_function

import os
import sys
import types

import numpy as np

for _module in ('pyBigWig', 'pybedtools'):
    if _module not in sys.modules:
        try:
            __import__(_module)
        except ImportError:
            sys.modules[_module] = types.ModuleType(_module)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import deepmex
import positionscore as ps

FLANK = deepmex.FLANK_SIZE
random = np.random.RandomState(0)


def make_microexon(strand):
    """A Microexon with random flanks, bypassing the genome/bigwig readers."""
    microexon = deepmex.Microexon.__new__(deepmex.Microexon)
    microexon.chr, microexon.start, microexon.end = 'chrX', 31126642, 31126673
    microexon.strand = strand
    up_bases = random.randint(0, 4, FLANK)
    down_bases = random.randint(0, 4, FLANK)
    microexon.encode_dna_up = ps.MUTATION_VECTORS[up_bases].astype(float)
    microexon.encode_dna_down = ps.MUTATION_VECTORS[down_bases].astype(float)
    microexon.conservation_values_up = random.rand(1, FLANK, 1)
    microexon.conservation_values_down = random.rand(1, FLANK, 1)
    return microexon, up_bases, down_bases


class StandInModel(object):
    """Position and base sensitive score, so any mix up shows up as a diff."""

    def __init__(self):
        self.calls = []
        weights = np.arange(1, FLANK + 1)[:, None] * np.array([1.0, 3.0, 7.0, 11.0])[None, :]
        self.weights_up, self.weights_down = weights, weights * 0.5

    def predict(self, inputs, verbose=0):
        sequence_up, conservation_up, sequence_down, conservation_down = inputs
        self.calls.append([np.array(value) for value in inputs])
        score = ((sequence_up * self.weights_up).sum(axis=(1, 2))
                 + (sequence_down * self.weights_down).sum(axis=(1, 2))
                 + conservation_up.sum(axis=(1, 2)) * 0.13
                 + conservation_down.sum(axis=(1, 2)) * 0.29)
        return (score / 1e5).reshape(-1, 1)


def test_plus_strand_keeps_input_order():
    microexon, _, _ = make_microexon('+')
    model = StandInModel()
    microexon.predict_batch(model, microexon.encode_dna_up[None], microexon.encode_dna_down[None])
    sequence_up, conservation_up, sequence_down, conservation_down = model.calls[0]
    assert np.array_equal(sequence_up[0], microexon.encode_dna_up)
    assert np.array_equal(sequence_down[0], microexon.encode_dna_down)
    assert np.allclose(conservation_up[0], microexon.conservation_values_up[0])
    assert np.allclose(conservation_down[0], microexon.conservation_values_down[0])


def test_minus_strand_swaps_and_reverses_flanks():
    microexon, _, _ = make_microexon('-')
    model = StandInModel()
    microexon.predict_batch(model, microexon.encode_dna_up[None], microexon.encode_dna_down[None])
    sequence_up, conservation_up, sequence_down, conservation_down = model.calls[0]
    assert np.array_equal(sequence_up[0], microexon.encode_dna_down[::-1])
    assert np.array_equal(sequence_down[0], microexon.encode_dna_up[::-1])
    assert np.allclose(conservation_up[0], microexon.conservation_values_down[0])
    assert np.allclose(conservation_down[0], microexon.conservation_values_up[0])


def test_batch_matches_single_prediction():
    for strand in '+-':
        microexon, _, _ = make_microexon(strand)
        model = StandInModel()
        single = microexon.prediction(model)
        batch = microexon.predict_batch(
            model,
            np.repeat(microexon.encode_dna_up[None], 5, axis=0),
            np.repeat(microexon.encode_dna_down[None], 5, axis=0))
        assert np.allclose(batch, single), (strand, batch, single)


def test_screening_shape_and_wild_type_baseline():
    for strand in '+-':
        microexon, up_bases, down_bases = make_microexon(strand)
        model = StandInModel()
        matrix, _ = ps.position_score_matrix(microexon, model, batch_size=97)
        assert matrix.shape == (2 * FLANK, len(ps.MUTATION_BASES))
        wild_type = np.concatenate([up_bases, down_bases])
        assert np.allclose(matrix[np.arange(2 * FLANK), wild_type], 0)
        scored = sum(call[0].shape[0] for call in model.calls)
        assert scored == 2 * FLANK * len(ps.MUTATION_BASES) + 1, scored


def test_screening_matches_position_by_position_loop():
    """The batched screening must equal the notebook loop, variant by variant."""
    microexon, _, _ = make_microexon('+')
    model = StandInModel()
    matrix, _ = ps.position_score_matrix(microexon, model)

    sequence = np.concatenate([microexon.encode_dna_up, microexon.encode_dna_down])
    wild_type_score = microexon.predict_batch(
        model, microexon.encode_dna_up[None], microexon.encode_dna_down[None])[0]
    expected = np.zeros_like(matrix)
    for position in range(sequence.shape[0]):
        for base in range(len(ps.MUTATION_BASES)):
            mutated = sequence.copy()
            mutated[position] = ps.MUTATION_VECTORS[base]
            expected[position, base] = microexon.predict_batch(
                model, mutated[None, :FLANK], mutated[None, FLANK:])[0] - wild_type_score
    assert np.allclose(matrix, expected), np.abs(matrix - expected).max()


def test_transcript_orientation():
    matrix = np.arange(800).reshape(200, 4).astype(float)
    assert np.array_equal(ps.to_transcript_orientation(matrix, '+'), matrix)
    assert np.array_equal(ps.to_transcript_orientation(matrix, '-'), matrix[::-1])


def test_plot_matrix_layout():
    matrix = np.arange(800).reshape(200, 4).astype(float)
    plot_matrix = ps.build_plot_matrix(matrix, base_order='TGAC', gap=4)
    assert plot_matrix.shape == (4, 204)
    assert np.isnan(plot_matrix[:, FLANK:FLANK + 4]).all()
    assert not np.isnan(plot_matrix[:, :FLANK]).any()
    assert not np.isnan(plot_matrix[:, FLANK + 4:]).any()
    assert plot_matrix[0, 0] == matrix[0, ps.MUTATION_BASES.index('T')]
    assert plot_matrix[3, 0] == matrix[0, ps.MUTATION_BASES.index('C')]
    assert plot_matrix[0, FLANK + 4] == matrix[FLANK, ps.MUTATION_BASES.index('T')]


def test_tsv_positions(tmp_path='positionscore_test.tsv'):
    matrix = np.arange(800).reshape(200, 4).astype(float)
    ps.write_tsv(tmp_path, matrix, 'chrX', 31126642, 31126673, '+')
    try:
        lines = open(tmp_path).read().strip().split('\n')
        assert len(lines) == 201
        assert lines[1].split('\t')[4] == '-100'
        assert lines[100].split('\t')[4] == '-1'
        assert lines[101].split('\t')[4] == '1'
        assert lines[200].split('\t')[4] == '100'
        assert lines[1].split('\t')[6:] == ['0', '1', '2', '3']
    finally:
        os.remove(tmp_path)


def test_exon_parsing():
    assert ps.parse_exon('chrX:31126642:31126673:+') == ('chrX', 31126642, 31126673, '+')
    for bad in ('chrX:1:2', 'chrX-1-2-+'):
        try:
            ps.parse_exon(bad)
        except ValueError:
            continue
        raise AssertionError('{!r} should not parse'.format(bad))


if __name__ == '__main__':
    tests = sorted(name for name in dir() if name.startswith('test_'))
    for name in tests:
        globals()[name]()
        print('ok  {}'.format(name))
    print('\n{} checks passed'.format(len(tests)))
