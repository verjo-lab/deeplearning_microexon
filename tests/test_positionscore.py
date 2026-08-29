"""Checks for the mutational screening, with a stand in model.

Runs without tensorflow, without the reference files and without a GPU: the
CNN is replaced by a deterministic scoring function, so only the wiring is
under test.

    uv run pytest
"""

import numpy as np
import pytest

from deepmex.core import FLANK_SIZE, Microexon
from deepmex.impact import impact_table
from deepmex.positionscore import (
    MUTATION_BASES,
    MUTATION_VECTORS,
    build_plot_matrix,
    position_score_matrix,
    to_transcript_orientation,
    write_tsv,
)


class StandInModel:
    """Position and base sensitive score, so any mix up shows up as a diff."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []
        weights = np.arange(1, FLANK_SIZE + 1)[:, None] * np.array([1.0, 3.0, 7.0, 11.0])[None, :]
        self.weights_up, self.weights_down = weights, weights * 0.5

    def predict(self, inputs: list[np.ndarray], verbose: int = 0) -> np.ndarray:
        sequence_up, conservation_up, sequence_down, conservation_down = inputs
        self.calls.append([np.asarray(value) for value in inputs])
        score = (
            (sequence_up * self.weights_up).sum(axis=(1, 2))
            + (sequence_down * self.weights_down).sum(axis=(1, 2))
            + conservation_up.sum(axis=(1, 2)) * 0.13
            + conservation_down.sum(axis=(1, 2)) * 0.29
        )
        return (score / 1e5).reshape(-1, 1)


def make_microexon(strand: str, seed: int = 0) -> tuple[Microexon, np.ndarray, np.ndarray]:
    """A Microexon with random flanks, bypassing the genome/bigwig readers."""
    random = np.random.default_rng(seed)
    up_bases = random.integers(0, 4, FLANK_SIZE)
    down_bases = random.integers(0, 4, FLANK_SIZE)
    microexon = Microexon(
        chrom="chrX",
        start=31126642,
        end=31126673,
        strand=strand,
        encode_dna_up=MUTATION_VECTORS[up_bases].astype(float),
        encode_dna_down=MUTATION_VECTORS[down_bases].astype(float),
        conservation_up=random.random((1, FLANK_SIZE, 1)),
        conservation_down=random.random((1, FLANK_SIZE, 1)),
    )
    return microexon, up_bases, down_bases


def test_plus_strand_keeps_input_order():
    microexon, _, _ = make_microexon("+")
    model = StandInModel()
    microexon.predict_batch(model, microexon.encode_dna_up[None], microexon.encode_dna_down[None])

    sequence_up, conservation_up, sequence_down, conservation_down = model.calls[0]
    assert np.array_equal(sequence_up[0], microexon.encode_dna_up)
    assert np.array_equal(sequence_down[0], microexon.encode_dna_down)
    assert np.allclose(conservation_up[0], microexon.conservation_up[0])
    assert np.allclose(conservation_down[0], microexon.conservation_down[0])


def test_minus_strand_swaps_and_reverses_flanks():
    microexon, _, _ = make_microexon("-")
    model = StandInModel()
    microexon.predict_batch(model, microexon.encode_dna_up[None], microexon.encode_dna_down[None])

    sequence_up, conservation_up, sequence_down, conservation_down = model.calls[0]
    assert np.array_equal(sequence_up[0], microexon.encode_dna_down[::-1])
    assert np.array_equal(sequence_down[0], microexon.encode_dna_up[::-1])
    assert np.allclose(conservation_up[0], microexon.conservation_down[0])
    assert np.allclose(conservation_down[0], microexon.conservation_up[0])


@pytest.mark.parametrize("strand", ["+", "-"])
def test_batch_matches_single_prediction(strand):
    microexon, _, _ = make_microexon(strand)
    model = StandInModel()

    single = microexon.predict(model)
    batch = microexon.predict_batch(
        model,
        np.repeat(microexon.encode_dna_up[None], 5, axis=0),
        np.repeat(microexon.encode_dna_down[None], 5, axis=0),
    )

    assert np.allclose(batch, single)


@pytest.mark.parametrize("strand", ["+", "-"])
def test_screening_shape_and_wild_type_baseline(strand):
    microexon, up_bases, down_bases = make_microexon(strand)
    model = StandInModel()

    matrix, _ = position_score_matrix(microexon, model, batch_size=97)

    assert matrix.shape == (2 * FLANK_SIZE, len(MUTATION_BASES))
    wild_type = np.concatenate([up_bases, down_bases])
    assert np.allclose(matrix[np.arange(2 * FLANK_SIZE), wild_type], 0)
    scored = sum(call[0].shape[0] for call in model.calls)
    assert scored == 2 * FLANK_SIZE * len(MUTATION_BASES) + 1


def test_screening_matches_position_by_position_loop():
    """The batched screening must equal the notebook loop, variant by variant."""
    microexon, _, _ = make_microexon("+")
    model = StandInModel()
    matrix, _ = position_score_matrix(microexon, model)

    sequence = np.concatenate([microexon.encode_dna_up, microexon.encode_dna_down])
    wild_type_score = microexon.predict(model)
    expected = np.zeros_like(matrix)
    for position in range(sequence.shape[0]):
        for base in range(len(MUTATION_BASES)):
            mutated = sequence.copy()
            mutated[position] = MUTATION_VECTORS[base]
            expected[position, base] = (
                microexon.predict_batch(model, mutated[None, :FLANK_SIZE], mutated[None, FLANK_SIZE:])[0]
                - wild_type_score
            )

    assert np.allclose(matrix, expected)


def test_transcript_orientation():
    matrix = np.arange(800).reshape(200, 4).astype(float)
    assert np.array_equal(to_transcript_orientation(matrix, "+"), matrix)


def test_transcript_orientation_reverse_complements_the_minus_strand():
    """The row labelled A must carry the mutation to T of the genome."""
    matrix = np.arange(800).reshape(200, 4).astype(float)

    flipped = to_transcript_orientation(matrix, "-")

    for base, complement in (("A", "T"), ("C", "G"), ("T", "A"), ("G", "C")):
        assert np.array_equal(
            flipped[:, MUTATION_BASES.index(base)],
            matrix[::-1, MUTATION_BASES.index(complement)],
        )


def test_plot_matrix_layout():
    matrix = np.arange(800).reshape(200, 4).astype(float)

    plot_matrix = build_plot_matrix(matrix, base_order="TGAC", gap=4)

    assert plot_matrix.shape == (4, 204)
    assert np.isnan(plot_matrix[:, FLANK_SIZE : FLANK_SIZE + 4]).all()
    assert not np.isnan(plot_matrix[:, :FLANK_SIZE]).any()
    assert not np.isnan(plot_matrix[:, FLANK_SIZE + 4 :]).any()
    assert plot_matrix[0, 0] == matrix[0, MUTATION_BASES.index("T")]
    assert plot_matrix[3, 0] == matrix[0, MUTATION_BASES.index("C")]
    assert plot_matrix[0, FLANK_SIZE + 4] == matrix[FLANK_SIZE, MUTATION_BASES.index("T")]


def test_tsv_positions(tmp_path):
    matrix = np.arange(800).reshape(200, 4).astype(float)
    path = tmp_path / "positionscore.tsv"

    write_tsv(path, matrix, "chrX", 31126642, 31126673, "+")

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 201
    assert lines[1].split("\t")[4] == "-100"
    assert lines[101].split("\t")[4] == "1"
    assert lines[200].split("\t")[4] == "100"
    assert lines[1].split("\t")[6:] == ["0", "1", "2", "3"]


def test_impact_table_zeroes_the_wild_type():
    """The reference base of a position must be the one scoring exactly 0."""
    for strand in ("+", "-"):
        microexon, _, _ = make_microexon(strand)
        model = StandInModel()
        matrix, _ = position_score_matrix(microexon, model)
        matrix = to_transcript_orientation(matrix, strand)

        table = impact_table(matrix, microexon)

        assert len(table) == 2 * FLANK_SIZE
        for entry in table:
            assert entry.deltas[entry.wild_type] == 0.0, (strand, entry)


def test_impact_table_reports_genomic_positions():
    microexon, _, _ = make_microexon("+")
    model = StandInModel()
    matrix, _ = position_score_matrix(microexon, model)

    table = impact_table(to_transcript_orientation(matrix, "+"), microexon)
    by_relative = {entry.relative_position: entry for entry in table}

    # The flanks sit immediately outside the exon, in genomic order.
    assert by_relative[-1].genomic_position == microexon.start - 1
    assert by_relative[-100].genomic_position == microexon.start - 100
    assert by_relative[1].genomic_position == microexon.end + 2
    assert by_relative[100].genomic_position == microexon.end + 101


def test_impact_table_is_sorted_and_ranked_by_impact():
    microexon, _, _ = make_microexon("-")
    model = StandInModel()
    matrix, _ = position_score_matrix(microexon, model)

    table = impact_table(to_transcript_orientation(matrix, "-"), microexon)

    impacts = [entry.impact for entry in table]
    assert impacts == sorted(impacts, reverse=True)
    assert table[0].impact == max(impacts)
