"""PositionScore: synthetic mutational screening around a microexon.

Every base of the two 100 nt flanks of a microexon is replaced by each of the
four nucleotides and re-scored with the CNN. The resulting delta matrix
(mutant score - wild type score) is drawn as the 4 x 200 heatmap used in the
paper (Fig. 7A).

Sample command::

    uv run positionscore \\
        --model src/saved_model.hdf5 \\
        --genome src/data/hg38.fa \\
        --conservation src/data/hg38_cons.bw \\
        --exon chrX:31126642:31126673:+ \\
        --gene DMD \\
        --output DMD_positionscore.png

Use ``--demo`` to render the layout with synthetic scores, without a model or
the reference files.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deepmex.core import (
    FLANK_SIZE,
    CNNModel,
    Microexon,
    ReferenceFiles,
    Strand,
    load_cnn_model,
    parse_exon,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from deepmex.knockdown import KnockdownEvent

# Mutated bases, in the same order used by the training notebook.
MUTATION_BASES = ["A", "C", "T", "G"]
MUTATION_VECTORS = np.array(
    [
        [0, 0, 0, 1],  # A
        [0, 0, 1, 0],  # C
        [0, 1, 0, 0],  # T
        [1, 0, 0, 0],  # G
    ]
)

#: Watson-Crick complement, used to draw minus strand exons in transcript
#: orientation.
COMPLEMENT = {"A": "T", "C": "G", "T": "A", "G": "C"}

# Row order of the published figure, top to bottom.
FIGURE_BASE_ORDER = "TGAC"

EXON_COLOR = "#29ABE2"
FRAME_COLOR = "#2233AA"

#: Color scale limit. The published figure uses 0.015 on the 0-1 scale of the
#: training notebook; the scores here are on the 0-100 scale of the CLI.
DEFAULT_VMAX = 1.5


def position_score_matrix(
    microexon: Microexon,
    model: CNNModel,
    batch_size: int = 256,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Return the (200, 4) delta score matrix of the mutational screening.

    Rows are flank positions (0-99 upstream, 100-199 downstream, in genomic
    orientation), columns are the mutated bases in ``MUTATION_BASES`` order.
    Values are ``score(mutant) - score(wild type)`` on the 0-100 scale of the
    CNN; the wild type base of each position scores exactly 0.
    """
    wild_type = np.concatenate([microexon.encode_dna_up, microexon.encode_dna_down])
    n_positions = wild_type.shape[0]
    n_bases = len(MUTATION_BASES)

    variants = np.repeat(wild_type[np.newaxis, :, :], n_positions * n_bases, axis=0)
    for position in range(n_positions):
        for base_index in range(n_bases):
            variants[position * n_bases + base_index, position] = MUTATION_VECTORS[base_index]

    wild_type_score = microexon.predict_batch(
        model,
        microexon.encode_dna_up[np.newaxis],
        microexon.encode_dna_down[np.newaxis],
    )[0]

    scores = []
    for start in range(0, variants.shape[0], batch_size):
        chunk = variants[start : start + batch_size]
        scores.append(microexon.predict_batch(model, chunk[:, :FLANK_SIZE], chunk[:, FLANK_SIZE:]))
        if verbose:
            scored = min(start + batch_size, variants.shape[0])
            print(f"scored {scored}/{variants.shape[0]} variants", file=sys.stderr)

    deltas = np.concatenate(scores) - wild_type_score
    return deltas.reshape(n_positions, n_bases), float(wild_type_score)


def to_transcript_orientation(matrix: np.ndarray, strand: Strand) -> np.ndarray:
    """Reverse complement the screening so the figure reads 5' to 3'.

    On the minus strand the transcript runs against the genome, so the
    position axis is reversed and every base is replaced by its complement:
    column 0 is always -100 relative to the exon, and a row labelled ``A`` is
    the mutation to A *of the transcript*, which is a mutation to T of the
    genome. This is the orientation of the published figure.
    """
    if strand == "+":
        return matrix
    complement = [MUTATION_BASES.index(COMPLEMENT[base]) for base in MUTATION_BASES]
    return matrix[::-1][:, complement]


def build_plot_matrix(
    matrix: np.ndarray,
    base_order: str = FIGURE_BASE_ORDER,
    gap: int = 4,
) -> np.ndarray:
    """Arrange the delta matrix as (4, 200 + gap) for plotting.

    ``base_order`` sets the row order top to bottom; ``gap`` blank columns are
    inserted between the two flanks to leave room for the microexon.
    """
    row_index = [MUTATION_BASES.index(base) for base in base_order]
    heat = matrix[:, row_index].T
    blank = np.full((heat.shape[0], gap), np.nan)
    return np.hstack([heat[:, :FLANK_SIZE], blank, heat[:, FLANK_SIZE:]])


def draw_position_scores(
    figure: "Figure",
    matrix: np.ndarray,
    chrom: str,
    start: int,
    end: int,
    strand: Strand,
    gene: str | None = None,
    vmax: float = DEFAULT_VMAX,
    gap: int = 4,
    base_order: str = FIGURE_BASE_ORDER,
    panel_label: str | None = None,
    subtitle: str | None = None,
    band: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Draw panel A into ``figure``, inside the vertical ``band`` (y0, height).

    ``band`` lets the panel be stacked with others in a composite figure; it
    defaults to the whole figure.
    """
    import matplotlib  # noqa: PLC0415
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    band_y0, band_height = band

    def place(x: float, y: float, width: float, tall: float) -> tuple[float, float, float, float]:
        return (x, band_y0 + y * band_height, width, tall * band_height)

    plot_matrix = build_plot_matrix(matrix, base_order=base_order, gap=gap)
    n_columns = plot_matrix.shape[1]

    colormap = matplotlib.colormaps["RdBu_r"].copy()
    colormap.set_bad("white")

    heat_axes = figure.add_axes(place(0.08, 0.42, 0.80, 0.36))
    gene_axes = figure.add_axes(place(0.08, 0.16, 0.80, 0.16))
    bar_axes = figure.add_axes(place(0.91, 0.42, 0.015, 0.36))

    image = heat_axes.imshow(
        plot_matrix, aspect="auto", cmap=colormap, vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    heat_axes.set_xticks([])
    heat_axes.set_yticks(range(len(base_order)))
    heat_axes.set_yticklabels(list(base_order), fontsize=16)
    heat_axes.tick_params(axis="y", length=0, pad=8)
    for spine in heat_axes.spines.values():
        spine.set_edgecolor(FRAME_COLOR)
        spine.set_linewidth(1.6)

    color_bar = figure.colorbar(image, cax=bar_axes, ticks=np.linspace(-vmax, vmax, 7))
    color_bar.ax.set_title("Score", fontsize=13, fontweight="bold", pad=10)
    color_bar.ax.tick_params(labelsize=10)
    color_bar.outline.set_linewidth(0.6)

    # Gene diagram: the two flanking introns and the microexon in between.
    gene_axes.set_xlim(0, n_columns)
    gene_axes.set_ylim(-1, 1)
    gene_axes.axis("off")
    gene_axes.plot([0, n_columns], [0, 0], color=FRAME_COLOR, lw=2, solid_capstyle="butt")
    gene_axes.add_patch(Rectangle((FLANK_SIZE, -0.45), gap, 0.9, facecolor=EXON_COLOR, edgecolor=EXON_COLOR))

    # Leader lines tying the blank columns of the heatmap to the microexon.
    # Both axes share the same x span, so column numbers are already aligned.
    heat_box = heat_axes.get_position()
    gene_box = gene_axes.get_position()
    heat_bottom = -1 + 2 * (heat_box.y0 - gene_box.y0) / gene_box.height
    for column in (FLANK_SIZE, FLANK_SIZE + gap):
        gene_axes.plot([column, column], [0.45, heat_bottom], color=EXON_COLOR, lw=1, clip_on=False, zorder=0)
    gene_axes.text(0, -0.75, "-100", ha="left", va="top", fontsize=18)
    gene_axes.text(n_columns, -0.75, "+100", ha="right", va="top", fontsize=18)

    exon_length = end - start + 1
    coordinate = f"{chrom}:{start}-{end}"
    if gene:
        title = f"{gene} ({exon_length} nt)\n{coordinate}"
    else:
        title = f"{exon_length} nt microexon ({strand} strand)\n{coordinate}"
    heat_axes.set_title(title, fontsize=19, pad=14)

    if subtitle:
        figure.text(0.5, band_y0 + 0.03 * band_height, subtitle, ha="center", fontsize=10, color="#B00020")
    if panel_label:
        figure.text(0.01, band_y0 + 0.93 * band_height, panel_label, fontsize=26, fontweight="bold")


def plot_figure(
    matrix: np.ndarray,
    chrom: str,
    start: int,
    end: int,
    strand: Strand,
    knockdown: "KnockdownEvent | None" = None,
    knockdown_title: str | None = None,
    **panel_a: object,
) -> "Figure":
    """Compose the published figure: panel A alone, or panels A and B stacked."""
    from matplotlib.figure import Figure  # noqa: PLC0415

    if knockdown is None:
        figure = Figure(figsize=(14, 4.4))
        draw_position_scores(figure, matrix, chrom, start, end, strand, **panel_a)
        return figure

    from deepmex.knockdown import draw_knockdown  # noqa: PLC0415

    height_a, height_b = 4.4, 5.2
    figure = Figure(figsize=(14, height_a + height_b))
    fraction_b = height_b / (height_a + height_b)
    if not panel_a.get("panel_label"):
        panel_a["panel_label"] = "A"
    draw_position_scores(
        figure, matrix, chrom, start, end, strand, band=(fraction_b, 1 - fraction_b), **panel_a
    )
    draw_knockdown(
        figure,
        knockdown,
        band=(0.0, fraction_b),
        title=knockdown_title,
        panel_label="B",
    )
    return figure


def plot_position_scores(*args, **kwargs) -> "Figure":
    """Draw panel A alone and return the figure.

    Built without pyplot, so scoring many microexons in a loop does not pile
    up figures in the global pyplot registry.
    """
    from matplotlib.figure import Figure  # noqa: PLC0415

    figure = Figure(figsize=(14, 4.4))
    draw_position_scores(figure, *args, **kwargs)
    return figure


def write_tsv(
    path: str | Path,
    matrix: np.ndarray,
    chrom: str,
    start: int,
    end: int,
    strand: Strand,
) -> None:
    """Dump the delta matrix as a tab separated table."""
    positions = list(range(-FLANK_SIZE, 0)) + list(range(1, FLANK_SIZE + 1))
    header = "\t".join(["chr", "start", "end", "strand", "position", "flank", *MUTATION_BASES])
    rows = [header]
    for index, position in enumerate(positions):
        flank = "upstream" if position < 0 else "downstream"
        scores = "\t".join(f"{value:.6g}" for value in matrix[index])
        rows.append(f"{chrom}\t{start}\t{end}\t{strand}\t{position}\t{flank}\t{scores}")
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def demo_matrix(seed: int = 7) -> np.ndarray:
    """Synthetic scores, for checking the layout without model or genome."""
    random = np.random.default_rng(seed)
    positions = np.arange(2 * FLANK_SIZE)
    # Same envelope as the notebook, scaled to the 0-100 scores of the CLI.
    envelope = 100 * (0.004 + 0.008 * np.exp(-((positions - 60) ** 2) / 600.0))
    envelope += 100 * 0.006 * np.exp(-((positions - 150) ** 2) / 2000.0)
    return random.normal(0, 1, (2 * FLANK_SIZE, len(MUTATION_BASES))) * envelope[:, None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="positionscore",
        description="Mutational screening heatmap (PositionScore) of a microexon.",
    )
    parser.add_argument("--exon", required=True, help="exon coordinate, i.e. chrX:31126642:31126673:+")
    parser.add_argument("--model", help="CNN model file (src/saved_model.hdf5)")
    parser.add_argument("--genome", help="HG38 genome fasta file")
    parser.add_argument("--conservation", help="HG38 conservation bigwig file")
    parser.add_argument("--gene", help="gene name printed in the title")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("positionscore.png"),
        help="output figure; the extension picks the format (png, pdf, svg, tif)",
    )
    parser.add_argument("--tsv", type=Path, help="also write the delta scores to this tab separated file")
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX,
        help=f"color scale limit, the scale runs from -vmax to +vmax (default: {DEFAULT_VMAX})",
    )
    parser.add_argument(
        "--gap", type=int, default=4, help="blank columns drawn between the flanks (default: 4)"
    )
    parser.add_argument(
        "--base-order",
        default=FIGURE_BASE_ORDER,
        help="heatmap row order, top to bottom (default: TGAC)",
    )
    parser.add_argument("--panel-label", help="panel letter drawn on the top left corner")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="variants scored per model call (default: 256)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="output resolution (default: 300)")
    parser.add_argument(
        "--demo", action="store_true", help="render synthetic scores instead of running the model"
    )
    parser.add_argument("--quiet", action="store_true", help="do not report progress")

    panel_b = parser.add_argument_group("panel B: inclusion under a knock-down")
    panel_b.add_argument("--knockdown", type=Path, help="vast-tools INCLUSION_LEVELS_FULL table")
    panel_b.add_argument("--event", help="vast-tools event id, i.e. HsaEX0019952")
    panel_b.add_argument(
        "--group",
        action="append",
        metavar="NAME=SAMPLE,SAMPLE",
        help="a group of samples, i.e. --group shRNA=s1,s2 --group Control=c1,c2 "
        "(give it twice: the treated group first)",
    )
    panel_b.add_argument("--knockdown-title", help='title of panel B, i.e. "PTBP1 knock-down in HepG2 cells"')
    return parser


def parse_groups(values: list[str]) -> dict[str, list[str]]:
    """Parse the ``--group NAME=SAMPLE,SAMPLE`` options, keeping their order."""
    groups: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--group must look like NAME=SAMPLE,SAMPLE, got {value!r}")
        name, samples = value.split("=", 1)
        parsed = [sample.strip() for sample in samples.split(",") if sample.strip()]
        if not parsed:
            raise ValueError(f"--group {name!r} lists no samples")
        groups[name.strip()] = parsed
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if sorted(args.base_order) != sorted(MUTATION_BASES):
        parser.error(f"--base-order must be a permutation of ACGT, got {args.base_order!r}")

    try:
        chrom, start, end, strand = parse_exon(args.exon)
    except ValueError as error:
        parser.error(str(error))

    subtitle = None
    if args.demo:
        matrix = demo_matrix()
        subtitle = "DEMO: synthetic scores, not a model prediction"
    else:
        missing = [name for name in ("model", "genome", "conservation") if getattr(args, name) is None]
        if missing:
            parser.error("--{} required (or use --demo)".format(", --".join(missing)))

        try:
            references = ReferenceFiles.from_paths(args.genome, args.conservation)
        except FileNotFoundError as error:
            parser.error(str(error))

        if not args.quiet:
            print(f"loading model {args.model}", file=sys.stderr)
        model = load_cnn_model(args.model)
        microexon = Microexon.from_reference(chrom, start, end, strand, references)
        matrix, wild_type_score = position_score_matrix(
            microexon, model, batch_size=args.batch_size, verbose=not args.quiet
        )
        print(f"Microexon prediction: {wild_type_score}")

    matrix = to_transcript_orientation(matrix, strand)

    knockdown = None
    if args.knockdown or args.event or args.group:
        missing = [name for name in ("knockdown", "event", "group") if not getattr(args, name)]
        if missing:
            parser.error("--{} required to draw panel B".format(", --".join(missing)))
        from deepmex.knockdown import read_vast_tools  # noqa: PLC0415

        try:
            knockdown = read_vast_tools(args.knockdown, args.event, parse_groups(args.group))
        except (ValueError, OSError) as error:
            parser.error(str(error))

    if args.tsv:
        write_tsv(args.tsv, matrix, chrom, start, end, strand)
        print(f"scores written to {args.tsv}")

    figure = plot_figure(
        matrix,
        chrom,
        start,
        end,
        strand,
        knockdown=knockdown,
        knockdown_title=args.knockdown_title,
        gene=args.gene,
        vmax=args.vmax,
        gap=args.gap,
        base_order=args.base_order,
        panel_label=args.panel_label,
        subtitle=subtitle,
    )
    # An uncompressed 300 dpi TIFF of this panel is ~20 MB; LZW is lossless
    # and is what journals expect.
    save_kwargs = {}
    if args.output.suffix.lower() in (".tif", ".tiff"):
        save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    figure.savefig(args.output, dpi=args.dpi, **save_kwargs)
    print(f"figure written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
