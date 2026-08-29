"""Read the flanks of an exon from the UCSC API instead of local files.

The reference genome and the conservation bigwig are about 6 GB together. For
a handful of microexons the REST API of the genome browser answers the same
questions over the network, which also fills in the strand and the gene name
from the RefSeq annotation.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

from deepmex.core import FLANK_SIZE, Microexon, Strand, encode_sequence, flank_intervals

API = "https://api.genome.ucsc.edu/getData"
GENOME = "hg38"
CONSERVATION_TRACK = "phastCons100way"
ANNOTATION_TRACK = "ncbiRefSeqCurated"
TIMEOUT = 120


class UCSCError(RuntimeError):
    """The API could not be reached, or answered something unusable."""


def fetch(url: str, attempts: int = 4) -> dict:
    """GET a JSON payload, retrying a few times on a network hiccup."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == attempts - 1:
                raise UCSCError(f"{url} failed after {attempts} attempts: {error}") from error
            time.sleep(2 * (attempt + 1))
    raise UCSCError("unreachable")


def sequence(chrom: str, start: int, end: int, genome: str = GENOME) -> str:
    """Genomic sequence of the half open interval ``[start, end)``."""
    payload = fetch(f"{API}/sequence?genome={genome};chrom={chrom};start={start};end={end}")
    dna = payload.get("dna")
    if not dna:
        raise UCSCError(f"no sequence for {chrom}:{start}-{end}")
    return dna


def _intervals(payload: dict, track: str) -> list[dict]:
    """The API returns a list, or a dict keyed by chromosome for wide queries."""
    values = payload.get(track)
    if isinstance(values, dict):
        return [interval for chunk in values.values() for interval in chunk]
    return values or []


def conservation(
    chrom: str,
    start: int,
    end: int,
    genome: str = GENOME,
    track: str = CONSERVATION_TRACK,
) -> tuple[np.ndarray, int]:
    """phastCons of ``[start, end)`` as a (1, n, 1) array, and how many bases had data.

    Positions the track does not cover score 0, as they do in the bigwig.
    """
    payload = fetch(f"{API}/track?genome={genome};track={track};chrom={chrom};start={start};end={end}")
    values = np.zeros(end - start, dtype=np.float32)
    covered = 0
    for interval in _intervals(payload, track):
        for position in range(max(interval["start"], start), min(interval["end"], end)):
            values[position - start] = interval["value"]
            covered += 1
    return values.reshape(1, -1, 1), covered


@dataclass(frozen=True, slots=True)
class Annotation:
    """What the RefSeq track says about the exon."""

    strand: Strand | None
    gene: str | None
    transcripts: int


def annotate(chrom: str, start: int, end: int, genome: str = GENOME) -> Annotation:
    """Strand and gene name of the transcripts covering the exon.

    When transcripts disagree the most frequent value wins; with no transcript
    at all both come back as ``None``.
    """
    payload = fetch(
        f"{API}/track?genome={genome};track={ANNOTATION_TRACK};chrom={chrom};start={start};end={end}"
    )
    transcripts = _intervals(payload, ANNOTATION_TRACK)
    strands = [t["strand"] for t in transcripts if t.get("strand")]
    genes = [t["name2"] for t in transcripts if t.get("name2")]
    return Annotation(
        strand=max(set(strands), key=strands.count) if strands else None,
        gene=max(set(genes), key=genes.count) if genes else None,
        transcripts=len(transcripts),
    )


def fetch_microexon(
    chrom: str,
    start: int,
    end: int,
    strand: Strand,
    genome: str = GENOME,
) -> tuple[Microexon, int]:
    """Build a :class:`Microexon` from the API, and report the conservation coverage."""
    upstream, downstream = flank_intervals(start, end)
    conservation_up, covered_up = conservation(chrom, *upstream, genome=genome)
    conservation_down, covered_down = conservation(chrom, *downstream, genome=genome)
    microexon = Microexon(
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        encode_dna_up=encode_sequence(sequence(chrom, *upstream, genome=genome)),
        encode_dna_down=encode_sequence(sequence(chrom, *downstream, genome=genome)),
        conservation_up=conservation_up,
        conservation_down=conservation_down,
    )
    return microexon, covered_up + covered_down


def coverage_summary(covered: int) -> str:
    return f"{covered}/{2 * FLANK_SIZE} flank bases with conservation data"
