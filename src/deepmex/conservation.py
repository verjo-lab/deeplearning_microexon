"""Per base conservation of the flanking introns, from a bedGraph track.

Used to build the training set: the flanks of every exon are intersected with
the genome wide conservation track (``bedmap``, from BEDOPS) and the resulting
intervals are unpacked to one value per base.

Requires the ``gtf`` extra (pandas, tqdm) and ``bedtools``/``bedmap`` on PATH.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

UP, DOWN = "up", "down"

#: A conservation interval plus the flank it was mapped to (bed4 + bed4).
MAPPED_COLUMNS = 8


@dataclass(frozen=True, slots=True)
class Workspace:
    """Where the intermediate bed files of the pipeline are written."""

    directory: Path

    @property
    def flanks(self) -> Path:
        return self.directory / "flank.bed"

    @property
    def mapped(self) -> Path:
        return self.directory / "flanks_conservation.bed"

    @property
    def mapped_fixed(self) -> Path:
        return self.directory / "flanks_conservation_fixed.bed"

    @property
    def unpacked(self) -> Path:
        return self.directory / "conservation_per_base.bed"

    @property
    def intersected(self) -> Path:
        return self.directory / "flanks_conservation_per_base.bed"

    def prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)


def build_flank_bed(flank_regions: pd.DataFrame, size_sequence: int) -> str:
    """Return a bed string with the two flanks of every exon.

    ``flank_regions`` needs the columns chr, start, end and short_id, where
    short_id ends with ``_<start>_<end>`` of the exon itself.
    """
    lines = []
    for chrom, _, _, short_id in flank_regions[["chr", "start", "end", "short_id"]].itertuples(index=False):
        exon_start, exon_end = (int(value) for value in short_id.split("_")[1:3])
        for side in (UP, DOWN):
            if side == UP:
                end = exon_start - 1
                start = end - size_sequence
            else:
                start = exon_end + 1
                end = start + size_sequence
            lines.append(f"chr{chrom}\t{start}\t{end}\t{short_id}_{side}")
    return "\n".join(lines)


def map_conservation(conservation_bedgraph: Path, flanks_bed: Path, output: Path) -> Path:
    """Intersect the flanks with the conservation track using ``bedmap``."""
    with output.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                "bedmap",
                "--skip-unmapped",
                "--echo",
                "--echo-map",
                str(conservation_bedgraph),
                str(flanks_bed),
            ],
            stdout=handle,
            check=True,
        )
    return output


def split_multiple_hits(mapped_bed: Path, output: Path) -> Path:
    """One line per (conservation interval, flank) pair.

    ``bedmap --echo-map`` packs every hit of a region on a single line,
    separated by ``;`` and ``|``.
    """
    rows: list[str] = []
    for line in mapped_bed.read_text(encoding="utf-8").splitlines():
        if ";" in line:
            fields = line.replace("|", ";").split(";")
            rows.extend(f"{fields[0]}\t{hit}" for hit in fields[1:])
        else:
            rows.append(line.replace("|", "\t"))
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def unpack_to_single_bases(conservation: pd.DataFrame) -> str:
    """Expand every conservation interval into one bed line per base."""
    lines: list[str] = []
    for row in tqdm(conservation.itertuples(index=False), desc="unpacking conservation"):
        if len(row) != MAPPED_COLUMNS:
            continue
        chrom, start, end, score = row[0], int(row[1]), int(row[2]), float(row[3])
        rounded = round(score, 1)
        lines.extend(
            f"{chrom}\t{start + offset}\t{start + offset + 1}\t{rounded}" for offset in range(end - start)
        )
    return "\n".join(lines)


def exon_key(flank_name: str) -> str:
    """Drop the ``_up``/``_down`` suffix a flank name carries."""
    return flank_name.removesuffix(f"_{UP}").removesuffix(f"_{DOWN}")


def get_conservation_hash(
    size_sequence: int,
    flanks_regions: pd.DataFrame,
    conservation_bedgraph: str | Path,
    workspace_directory: str | Path,
) -> dict[str, list[list[float]]]:
    """Return ``{exon_key: [upstream_values, downstream_values]}``.

    Both lists hold ``size_sequence`` per base conservation values, in genomic
    orientation.
    """
    from pybedtools import BedTool  # noqa: PLC0415  (optional dependency, see the genome extra)

    workspace = Workspace(Path(workspace_directory))
    workspace.prepare()

    print("sorting flanks...")
    BedTool(build_flank_bed(flanks_regions, size_sequence), from_string=True).sort().saveas(
        str(workspace.flanks)
    )

    print("intersecting with the conservation track (this takes a while)...")
    map_conservation(Path(conservation_bedgraph), workspace.flanks, workspace.mapped)
    split_multiple_hits(workspace.mapped, workspace.mapped_fixed)

    conservation = pd.read_csv(workspace.mapped_fixed, sep="\t", header=None)
    per_base = BedTool(unpack_to_single_bases(conservation), from_string=True).sort()
    per_base.saveas(str(workspace.unpacked))

    intersected = per_base.intersect(str(workspace.flanks), wb=True, wa=True)
    intersected.saveas(str(workspace.intersected))
    print(f"{len(intersected)} intersected intervals")

    print("removing duplicates...")
    per_base_df = pd.read_csv(workspace.intersected, sep="\t", header=None).drop_duplicates()

    by_exon: dict[str, list] = {}
    for row in tqdm(per_base_df.itertuples(index=False), desc="grouping by exon"):
        if len(row) == MAPPED_COLUMNS:
            by_exon.setdefault(exon_key(row[-1]), []).append(row)

    return {
        key: [
            [float(value[3]) for value in values[:size_sequence]],
            [float(value[3]) for value in values[size_sequence:]],
        ]
        for key, values in tqdm(by_exon.items(), desc="splitting flanks")
    }
