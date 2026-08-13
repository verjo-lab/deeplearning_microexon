"""Feature extraction and CNN scoring of candidate microexons.

A microexon is scored from its two 100 nt flanking introns: the one hot
encoded sequence and the phastCons conservation vector of each flank are fed
to the CNN as four separate inputs.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

import numpy as np

FLANK_SIZE = 100

type Strand = Literal["+", "-"]

#: Colon separated fields of a coordinate: ``chrom:start:end:strand`` has four,
#: ``chrom:start-end:strand`` three and ``chrom:start-end`` two.
COORDINATE_FIELDS = 4
COORDINATE_FIELDS_WITH_STRAND = 3
COORDINATE_FIELDS_BARE = 2

#: One hot encoding used when the model was trained (see the training notebook).
BASE_ENCODING: dict[str, tuple[int, int, int, int]] = {
    "A": (0, 0, 0, 1),
    "C": (0, 0, 1, 0),
    "T": (0, 1, 0, 0),
    "G": (1, 0, 0, 0),
    "N": (0, 0, 0, 0),
}


class CNNModel(Protocol):
    """The slice of the Keras model API used here."""

    def predict(self, inputs: Sequence[np.ndarray], verbose: int = 0) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ReferenceFiles:
    """The genome fasta and the conservation bigwig a microexon is read from."""

    genome: Path
    conservation: Path

    def __post_init__(self) -> None:
        for label, path in (("genome", self.genome), ("conservation", self.conservation)):
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

    @classmethod
    def from_paths(cls, genome: str | Path, conservation: str | Path) -> Self:
        return cls(Path(genome), Path(conservation))


def load_cnn_model(model_file: str | Path, compile_model: bool = False) -> CNNModel:
    """Load the trained Keras model, keeping the backend quiet.

    ``saved_model.hdf5`` was written by Keras 2.3, whose optimizer config
    (``lr=...``) no longer deserializes on Keras 3. Scoring does not need the
    optimizer, so the model is loaded uncompiled unless asked otherwise.
    """
    import tensorflow as tf  # noqa: PLC0415  (optional, heavy import)

    tf.get_logger().setLevel("ERROR")
    from keras.models import load_model  # noqa: PLC0415

    return load_model(str(model_file), compile=compile_model)


def read_conservation(conservation_file: str | Path, chrom: str, start: int, end: int) -> np.ndarray:
    """Return the phastCons values of ``[start, end)`` as a (1, n, 1) array."""
    import pyBigWig  # noqa: PLC0415  (optional dependency, see the genome extra)

    with pyBigWig.open(str(conservation_file)) as big_wig:
        values = big_wig.values(chrom, start, end)
    if values is None:
        raise ValueError(f"no conservation values for {chrom}:{start}-{end} in {conservation_file}")
    return np.asarray(values, dtype=np.float32).reshape(1, -1, 1)


def read_sequence(genome_file: str | Path, chrom: str, start: int, end: int) -> str:
    """Return the genomic sequence of ``[start, end)`` from the fasta file."""
    import pybedtools  # noqa: PLC0415  (optional dependency, see the genome extra)

    interval = pybedtools.BedTool(f"{chrom} {start} {end}", from_string=True)
    return interval.seq((chrom, start, end), str(genome_file))


def encode_sequence(sequence: Iterable[str]) -> np.ndarray:
    """One hot encode a DNA sequence as a (n, 4) float array."""
    try:
        encoded = [BASE_ENCODING[base.upper()] for base in sequence]
    except KeyError as error:
        raise ValueError(f"unknown nucleotide {error.args[0]!r} in sequence") from error
    return np.asarray(encoded, dtype=np.float32)


def flank_intervals(start: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Half open intervals of the two flanks of an exon, in genomic order.

    Single source of truth for the flank arithmetic: everything that reads a
    flank goes through here, so the reference files, the UCSC API and the
    position table can never drift apart.
    """
    upstream = (start - 1 - FLANK_SIZE, start - 1)
    downstream = (end + 1, end + 1 + FLANK_SIZE)
    return upstream, downstream


def parse_coordinate(text: str) -> tuple[str, int, int, Strand | None]:
    """Parse an exon coordinate, with or without the strand.

    Accepts ``chrX:31126642-31126673``, ``chrX:31126642-31126673:-`` and the
    older ``chrX:31126642:31126673:-``. The strand is ``None`` when absent.
    """
    fields = text.split(":")
    strand: str | None = None
    if len(fields) == COORDINATE_FIELDS_BARE:
        chrom, span = fields
    elif len(fields) == COORDINATE_FIELDS_WITH_STRAND and "-" in fields[1]:
        chrom, span, strand = fields
    elif len(fields) == COORDINATE_FIELDS_WITH_STRAND:
        chrom, first, second = fields
        span = f"{first}-{second}"
    elif len(fields) == COORDINATE_FIELDS:
        chrom, first, second, strand = fields
        span = f"{first}-{second}"
    else:
        raise ValueError(f"coordinate must look like chrX:31126642-31126673, got {text!r}")

    if strand is not None and strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")
    bounds = span.split("-")
    if len(bounds) != COORDINATE_FIELDS_BARE:
        raise ValueError(f"coordinate must look like chrX:31126642-31126673, got {text!r}")
    try:
        start, end = int(bounds[0]), int(bounds[1])
    except ValueError as error:
        raise ValueError(f"start and end must be integers, got {text!r}") from error
    if end < start:
        raise ValueError(f"end must not be before start, got {text!r}")
    return chrom, start, end, strand


def parse_exon(exon: str) -> tuple[str, int, int, Strand]:
    """Parse a coordinate that must carry the strand."""
    chrom, start, end, strand = parse_coordinate(exon)
    if strand is None:
        raise ValueError(f"exon must carry the strand, i.e. chrX:31126642-31126673:-, got {exon!r}")
    return chrom, start, end, strand


@dataclass(slots=True)
class Microexon:
    """A candidate microexon together with the features of its two flanks.

    ``encode_dna_*`` are (100, 4) one hot arrays and ``conservation_*`` are
    (1, 100, 1) arrays, both in genomic orientation. Use
    :meth:`from_reference` to build one from the reference files, or the
    constructor directly when the features come from somewhere else.
    """

    chrom: str
    start: int
    end: int
    strand: Strand
    encode_dna_up: np.ndarray
    encode_dna_down: np.ndarray
    conservation_up: np.ndarray
    conservation_down: np.ndarray

    def __post_init__(self) -> None:
        if self.strand not in ("+", "-"):
            raise ValueError(f"strand must be '+' or '-', got {self.strand!r}")

    @classmethod
    def from_reference(
        cls,
        chrom: str,
        start: int,
        end: int,
        strand: Strand,
        references: ReferenceFiles,
    ) -> Self:
        """Read the flank features of an exon from the genome and the bigwig."""
        upstream, downstream = flank_intervals(start, end)
        return cls(
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            encode_dna_up=encode_sequence(read_sequence(references.genome, chrom, *upstream)),
            encode_dna_down=encode_sequence(read_sequence(references.genome, chrom, *downstream)),
            conservation_up=read_conservation(references.conservation, chrom, *upstream),
            conservation_down=read_conservation(references.conservation, chrom, *downstream),
        )

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def coordinate(self) -> str:
        return f"{self.chrom}:{self.start}:{self.end}:{self.strand}"

    def predict_batch(
        self,
        model: CNNModel,
        encode_dna_up: np.ndarray,
        encode_dna_down: np.ndarray,
        verbose: int = 0,
    ) -> np.ndarray:
        """Score a batch of sequence variants of this microexon.

        ``encode_dna_up``/``encode_dna_down`` are (n, 100, 4) arrays; the
        conservation vectors of the microexon are broadcast to the batch. On
        the minus strand the flanks are swapped and the sequences are read
        backwards, as when the model was trained.
        """
        encode_dna_up = np.asarray(encode_dna_up).reshape(-1, FLANK_SIZE, 4)
        encode_dna_down = np.asarray(encode_dna_down).reshape(-1, FLANK_SIZE, 4)
        batch_size = encode_dna_up.shape[0]
        conservation_up = np.repeat(self.conservation_up.reshape(1, FLANK_SIZE, 1), batch_size, axis=0)
        conservation_down = np.repeat(self.conservation_down.reshape(1, FLANK_SIZE, 1), batch_size, axis=0)

        if self.strand == "+":
            inputs = [encode_dna_up, conservation_up, encode_dna_down, conservation_down]
        else:
            inputs = [
                encode_dna_down[:, ::-1],
                conservation_down,
                encode_dna_up[:, ::-1],
                conservation_up,
            ]

        return model.predict(inputs, verbose=verbose)[:, 0] * 100

    def predict(self, model: CNNModel) -> float:
        """Score the wild type microexon, from 0 to 100 (> 50 is a microexon)."""
        scores = self.predict_batch(model, self.encode_dna_up[np.newaxis], self.encode_dna_down[np.newaxis])
        return float(scores[0])
