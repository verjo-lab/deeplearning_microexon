"""Rank the flank bases by how much the model reacts to mutating them.

The screening gives a delta for every (position, base) pair. Collapsing the
three substitutions of a position into one number says which bases of the two
flanks the CNN actually depends on, and where they are in the genome.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepmex.core import FLANK_SIZE, Microexon, flank_intervals
from deepmex.positionscore import COMPLEMENT, MUTATION_BASES, MUTATION_VECTORS

#: Where the 1 sits in the one hot vector of each base. The encoding is not the
#: identity -- A is ``[0, 0, 0, 1]`` -- so the argmax of an encoded base is not
#: its index in ``MUTATION_BASES``.
ONE_HOT_POSITION = {
    int(vector.argmax()): base for vector, base in zip(MUTATION_VECTORS, MUTATION_BASES, strict=True)
}

TSV_COLUMNS = (
    "rank",
    "chr",
    "genomic_position",
    "relative_position",
    "flank",
    "wild_type",
    "impact",
    "mean_abs_delta",
    *(f"delta_{base}" for base in MUTATION_BASES),
)


@dataclass(frozen=True, slots=True)
class PositionImpact:
    """One flank position, with the deltas of the three substitutions."""

    chrom: str
    genomic_position: int
    relative_position: int
    flank: str
    wild_type: str
    deltas: dict[str, float]

    @property
    def substitutions(self) -> list[float]:
        """The three deltas that actually change the base."""
        return [value for base, value in self.deltas.items() if base != self.wild_type]

    @property
    def impact(self) -> float:
        """Largest absolute change the position can produce."""
        return max(abs(value) for value in self.substitutions)

    @property
    def mean_abs_delta(self) -> float:
        return float(np.mean([abs(value) for value in self.substitutions]))


def genomic_positions(start: int, end: int) -> np.ndarray:
    """1-based genomic coordinate of every row of the screening, in genomic order."""
    (upstream_start, _), (downstream_start, _) = flank_intervals(start, end)
    upstream = np.arange(FLANK_SIZE) + upstream_start + 1
    downstream = np.arange(FLANK_SIZE) + downstream_start + 1
    return np.concatenate([upstream, downstream])


def wild_type_bases(microexon: Microexon) -> list[str]:
    """Reference base of every row of the screening, in genomic order.

    An unknown base encodes as all zeros and comes back as ``N``.
    """
    encoded = np.concatenate([microexon.encode_dna_up, microexon.encode_dna_down])
    return ["N" if row.sum() == 0 else ONE_HOT_POSITION[int(row.argmax())] for row in encoded]


def impact_table(matrix: np.ndarray, microexon: Microexon) -> list[PositionImpact]:
    """Rank the positions of a transcript oriented screening by impact.

    ``matrix`` is what :func:`~deepmex.positionscore.to_transcript_orientation`
    returns, so on the minus strand its rows and bases are already reverse
    complemented; the coordinates and the reference bases are flipped here to
    match, and every row still carries its genomic position.
    """
    positions = genomic_positions(microexon.start, microexon.end)
    bases = wild_type_bases(microexon)
    if microexon.strand == "-":
        positions = positions[::-1]
        bases = [COMPLEMENT.get(base, base) for base in reversed(bases)]

    relative = list(range(-FLANK_SIZE, 0)) + list(range(1, FLANK_SIZE + 1))
    table = [
        PositionImpact(
            chrom=microexon.chrom,
            genomic_position=int(positions[row]),
            relative_position=relative[row],
            flank="upstream" if relative[row] < 0 else "downstream",
            wild_type=bases[row],
            deltas={base: float(matrix[row, index]) for index, base in enumerate(MUTATION_BASES)},
        )
        for row in range(matrix.shape[0])
    ]
    return sorted(table, key=lambda entry: entry.impact, reverse=True)


def write_impact_tsv(path: str | Path, table: list[PositionImpact]) -> None:
    """Write the ranking as a tab separated table, most impactful first."""
    rows = ["\t".join(TSV_COLUMNS)]
    for rank, entry in enumerate(table, start=1):
        deltas = "\t".join(f"{entry.deltas[base]:.6g}" for base in MUTATION_BASES)
        rows.append(
            f"{rank}\t{entry.chrom}\t{entry.genomic_position}\t{entry.relative_position}\t"
            f"{entry.flank}\t{entry.wild_type}\t{entry.impact:.6g}\t{entry.mean_abs_delta:.6g}\t{deltas}"
        )
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def format_top(table: list[PositionImpact], count: int = 15) -> str:
    """A readable block with the most impactful positions, for the terminal."""
    header = f"{'#':>3}  {'position':>12}  {'rel':>5}  {'wt':>2}  {'impact':>8}   deltas"
    lines = [header, "-" * len(header)]
    for rank, entry in enumerate(table[:count], start=1):
        deltas = "  ".join(
            f"{base}{entry.deltas[base]:+.2f}" for base in MUTATION_BASES if base != entry.wild_type
        )
        lines.append(
            f"{rank:>3}  {entry.genomic_position:>12}  {entry.relative_position:>+5}  "
            f"{entry.wild_type:>2}  {entry.impact:>8.3f}   {deltas}"
        )
    return "\n".join(lines)
