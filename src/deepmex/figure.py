"""One command from a coordinate to the figure and the impact table.

    deepmex-figure chrX:31126642-31126673

Strand and gene name come from the RefSeq annotation, and the flanks from the
UCSC API, so nothing has to be downloaded first. Point ``--genome`` and
``--conservation`` at local files to use those instead.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepmex.core import Microexon, ReferenceFiles, Strand, load_cnn_model, parse_coordinate
from deepmex.impact import format_top, impact_table, write_impact_tsv
from deepmex.positionscore import (
    DEFAULT_VMAX,
    FIGURE_BASE_ORDER,
    plot_figure,
    position_score_matrix,
    to_transcript_orientation,
)

DEFAULT_MODEL = Path("src/saved_model.hdf5")
TOP_POSITIONS = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepmex-figure",
        description="Score a microexon, draw the figure and rank the flank bases by impact.",
        epilog="The strand is read from RefSeq when the coordinate does not carry one.",
    )
    parser.add_argument("coordinate", help="exon coordinate, i.e. chrX:31126642-31126673")
    parser.add_argument("--strand", choices=["+", "-"], help="override the annotated strand")
    parser.add_argument("--gene", help="override the annotated gene name")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help=f"CNN model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--genome", type=Path, help="local genome fasta; default is the UCSC API")
    parser.add_argument(
        "--conservation", type=Path, help="local conservation bigwig; default is the UCSC API"
    )
    parser.add_argument("--output", type=Path, help="output figure (default: <name>.png)")
    parser.add_argument("--table", type=Path, help="output ranking (default: <name>_impact.tsv)")
    parser.add_argument(
        "--top", type=int, default=TOP_POSITIONS, help=f"rows printed (default: {TOP_POSITIONS})"
    )
    parser.add_argument(
        "--vmax", type=float, default=DEFAULT_VMAX, help=f"color scale (default: {DEFAULT_VMAX})"
    )
    parser.add_argument(
        "--base-order", default=FIGURE_BASE_ORDER, help=f"row order (default: {FIGURE_BASE_ORDER})"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="variants per model call (default: 256)")
    parser.add_argument("--dpi", type=int, default=300, help="output resolution (default: 300)")
    parser.add_argument("--quiet", action="store_true", help="do not report progress")

    panel_b = parser.add_argument_group("panel B: inclusion under a knock-down")
    panel_b.add_argument("--knockdown", type=Path, help="vast-tools INCLUSION_LEVELS_FULL table")
    panel_b.add_argument("--event", help="vast-tools event id, i.e. HsaEX0019952")
    panel_b.add_argument(
        "--group",
        action="append",
        metavar="NAME=SAMPLE,SAMPLE",
        help="a group of samples, given twice, the treated group first",
    )
    panel_b.add_argument("--knockdown-title", help="title of panel B")
    return parser


def resolve_strand_and_gene(
    parser: argparse.ArgumentParser,
    chrom: str,
    start: int,
    end: int,
    coordinate_strand: Strand | None,
    args: argparse.Namespace,
) -> tuple[Strand, str | None]:
    """Take the strand and gene from the options, the coordinate or RefSeq."""
    strand = args.strand or coordinate_strand
    gene = args.gene
    if strand and gene:
        return strand, gene

    from deepmex.ucsc import UCSCError, annotate  # noqa: PLC0415

    try:
        annotation = annotate(chrom, start, end)
    except UCSCError as error:
        if not strand:
            parser.error(f"could not read the strand from RefSeq ({error}); pass --strand")
        return strand, gene

    strand = strand or annotation.strand
    if not strand:
        parser.error(
            f"no RefSeq transcript covers {chrom}:{start}-{end}, so the strand is unknown; pass --strand"
        )
    if not args.quiet:
        print(
            f"RefSeq: strand {annotation.strand or '?'}, gene {annotation.gene or '?'} "
            f"({annotation.transcripts} transcripts)",
            file=sys.stderr,
        )
    return strand, gene or annotation.gene


def load_microexon(
    parser: argparse.ArgumentParser,
    chrom: str,
    start: int,
    end: int,
    strand: Strand,
    args: argparse.Namespace,
) -> Microexon:
    """Read the flanks from the local files when given, otherwise from the API."""
    if bool(args.genome) != bool(args.conservation):
        parser.error("--genome and --conservation go together")

    if args.genome:
        try:
            references = ReferenceFiles.from_paths(args.genome, args.conservation)
        except FileNotFoundError as error:
            parser.error(str(error))
        return Microexon.from_reference(chrom, start, end, strand, references)

    from deepmex.ucsc import UCSCError, coverage_summary, fetch_microexon  # noqa: PLC0415

    if not args.quiet:
        print(f"fetching the flanks of {chrom}:{start}-{end} from the UCSC API", file=sys.stderr)
    try:
        microexon, covered = fetch_microexon(chrom, start, end, strand)
    except (UCSCError, ValueError) as error:
        parser.error(str(error))
    if not args.quiet:
        print(coverage_summary(covered), file=sys.stderr)
    return microexon


def load_knockdown(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not (args.knockdown or args.event or args.group):
        return None
    missing = [name for name in ("knockdown", "event", "group") if not getattr(args, name)]
    if missing:
        parser.error("--{} required to draw panel B".format(", --".join(missing)))

    from deepmex.knockdown import read_vast_tools  # noqa: PLC0415
    from deepmex.positionscore import parse_groups  # noqa: PLC0415

    try:
        return read_vast_tools(args.knockdown, args.event, parse_groups(args.group))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        chrom, start, end, coordinate_strand = parse_coordinate(args.coordinate)
    except ValueError as error:
        parser.error(str(error))

    strand, gene = resolve_strand_and_gene(parser, chrom, start, end, coordinate_strand, args)
    stem = Path(f"{gene}_{chrom}_{start}_{end}" if gene else f"{chrom}_{start}_{end}")
    output = args.output or stem.with_suffix(".png")
    table_path = args.table or stem.with_name(f"{stem.name}_impact.tsv")

    knockdown = load_knockdown(parser, args)
    microexon = load_microexon(parser, chrom, start, end, strand, args)

    if not args.model.exists():
        parser.error(f"model file not found: {args.model}")
    if not args.quiet:
        print(f"loading model {args.model}", file=sys.stderr)
    model = load_cnn_model(args.model)

    matrix, wild_type_score = position_score_matrix(
        microexon, model, batch_size=args.batch_size, verbose=not args.quiet
    )
    matrix = to_transcript_orientation(matrix, strand)

    label = f"{gene} " if gene else ""
    print(f"{label}{chrom}:{start}-{end} ({strand}, {microexon.length} nt)")
    print(f"Microexon prediction: {wild_type_score:.3f}")

    table = impact_table(matrix, microexon)
    write_impact_tsv(table_path, table)
    if args.top:
        print(f"\nmost impactful flank bases (of {len(table)}):")
        print(format_top(table, args.top))

    figure = plot_figure(
        matrix,
        chrom,
        start,
        end,
        strand,
        knockdown=knockdown,
        knockdown_title=args.knockdown_title,
        gene=gene,
        vmax=args.vmax,
        base_order=args.base_order,
    )
    save_kwargs = {}
    if output.suffix.lower() in (".tif", ".tiff"):
        save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    figure.savefig(output, dpi=args.dpi, **save_kwargs)

    print(f"\nfigure written to {output}")
    print(f"ranking written to {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
