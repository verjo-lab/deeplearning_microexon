"""PositionScore: synthetic mutational screening around a microexon.

Every base of the two 100 nt flanks of a microexon is replaced by each of the
four nucleotides and re-scored with the CNN. The resulting delta matrix
(mutant score - wild type score) is drawn as the 4 x 200 heatmap used in the
paper (Fig. 7A).

Sample command:

    python src/positionscore.py \
        --model src/saved_model.hdf5 \
        --genome src/data/hg38.fa \
        --conservation src/data/hg38_cons.bw \
        --exon chrX:31126642:31126673:+ \
        --gene DMD \
        --output DMD_positionscore.png

Use ``--demo`` to render the layout with synthetic scores, without a model or
the reference files.
"""

from __future__ import print_function

import argparse
import sys

import numpy as np

FLANK_SIZE = 100

# Mutated bases, in the same order used by the training notebook.
MUTATION_BASES = ['A', 'C', 'T', 'G']
MUTATION_VECTORS = np.array([
    [0, 0, 0, 1],  # A
    [0, 0, 1, 0],  # C
    [0, 1, 0, 0],  # T
    [1, 0, 0, 0],  # G
])

# Row order of the published figure, top to bottom.
FIGURE_BASE_ORDER = 'TGAC'

EXON_COLOR = '#29ABE2'
FRAME_COLOR = '#2233AA'


def position_score_matrix(microexon, model, batch_size=256, verbose=False):
    """Return the (200, 4) delta score matrix of the mutational screening.

    Rows are flank positions (0-99 upstream, 100-199 downstream, in genomic
    orientation), columns are the mutated bases in ``MUTATION_BASES`` order.
    Values are ``score(mutant) - score(wild type)`` on the 0-100 scale of
    ``deepmex.py``; the wild type base of each position scores exactly 0.
    """
    wild_type = np.concatenate([microexon.encode_dna_up, microexon.encode_dna_down])
    n_positions = wild_type.shape[0]

    variants = np.repeat(wild_type[np.newaxis, :, :], n_positions * len(MUTATION_BASES), axis=0)
    for position in range(n_positions):
        for base_index in range(len(MUTATION_BASES)):
            variants[position * len(MUTATION_BASES) + base_index, position] = MUTATION_VECTORS[base_index]

    wild_type_score = microexon.predict_batch(
        model,
        microexon.encode_dna_up[np.newaxis],
        microexon.encode_dna_down[np.newaxis],
    )[0]

    scores = []
    for start in range(0, variants.shape[0], batch_size):
        chunk = variants[start:start + batch_size]
        scores.append(microexon.predict_batch(
            model, chunk[:, :FLANK_SIZE], chunk[:, FLANK_SIZE:],
        ))
        if verbose:
            print('scored {}/{} variants'.format(
                min(start + batch_size, variants.shape[0]), variants.shape[0]), file=sys.stderr)

    deltas = np.concatenate(scores) - wild_type_score
    return deltas.reshape(n_positions, len(MUTATION_BASES)), wild_type_score


def to_transcript_orientation(matrix, strand):
    """Flip the flanks so that column 0 is always -100 relative to the exon.

    Mirrors what ``Microexon.predict_batch`` does for minus strand exons: the
    position axis is reversed, the bases are not complemented.
    """
    if strand == '-':
        return matrix[::-1]
    return matrix


def build_plot_matrix(matrix, base_order=FIGURE_BASE_ORDER, gap=4):
    """Arrange the delta matrix as (4, 200 + gap) for plotting.

    ``base_order`` sets the row order top to bottom; ``gap`` blank columns are
    inserted between the two flanks to leave room for the microexon.
    """
    row_index = [MUTATION_BASES.index(base) for base in base_order]
    heat = matrix[:, row_index].T
    blank = np.full((heat.shape[0], gap), np.nan)
    return np.hstack([heat[:, :FLANK_SIZE], blank, heat[:, FLANK_SIZE:]])


def plot_position_scores(matrix, chrom, start, end, strand, gene=None,
                         vmax=0.015, gap=4, base_order=FIGURE_BASE_ORDER,
                         panel_label=None, subtitle=None):
    """Draw the Fig. 7A panel and return the matplotlib figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plot_matrix = build_plot_matrix(matrix, base_order=base_order, gap=gap)
    n_columns = plot_matrix.shape[1]

    colormap = plt.get_cmap('RdBu_r')
    colormap = colormap.copy() if hasattr(colormap, 'copy') else colormap
    colormap.set_bad('white')

    figure = plt.figure(figsize=(14, 4.4))
    heat_axes = figure.add_axes([0.08, 0.42, 0.80, 0.36])
    gene_axes = figure.add_axes([0.08, 0.16, 0.80, 0.16])
    bar_axes = figure.add_axes([0.91, 0.42, 0.015, 0.36])

    image = heat_axes.imshow(plot_matrix, aspect='auto', cmap=colormap,
                             vmin=-vmax, vmax=vmax, interpolation='nearest')
    heat_axes.set_xticks([])
    heat_axes.set_yticks(range(len(base_order)))
    heat_axes.set_yticklabels(list(base_order), fontsize=16)
    heat_axes.tick_params(axis='y', length=0, pad=8)
    for spine in heat_axes.spines.values():
        spine.set_edgecolor(FRAME_COLOR)
        spine.set_linewidth(1.6)

    color_bar = figure.colorbar(image, cax=bar_axes,
                                ticks=np.linspace(-vmax, vmax, 7))
    color_bar.ax.set_title('Score', fontsize=13, fontweight='bold', pad=10)
    color_bar.ax.tick_params(labelsize=10)
    color_bar.outline.set_linewidth(0.6)

    # Gene diagram: the two flanking introns and the microexon in between.
    gene_axes.set_xlim(0, n_columns)
    gene_axes.set_ylim(-1, 1)
    gene_axes.axis('off')
    gene_axes.plot([0, n_columns], [0, 0], color=FRAME_COLOR, lw=2, solid_capstyle='butt')
    gene_axes.add_patch(Rectangle((FLANK_SIZE, -0.45), gap, 0.9,
                                  facecolor=EXON_COLOR, edgecolor=EXON_COLOR))

    # Leader lines tying the blank columns of the heatmap to the microexon.
    # Both axes share the same x span, so column numbers are already aligned.
    heat_box = heat_axes.get_position()
    gene_box = gene_axes.get_position()
    heat_bottom = -1 + 2 * (heat_box.y0 - gene_box.y0) / gene_box.height
    for column in (FLANK_SIZE, FLANK_SIZE + gap):
        gene_axes.plot([column, column], [0.45, heat_bottom], color=EXON_COLOR,
                       lw=1, clip_on=False, zorder=0)
    gene_axes.text(0, -0.75, '-100', ha='left', va='top', fontsize=18)
    gene_axes.text(n_columns, -0.75, '+100', ha='right', va='top', fontsize=18)

    exon_length = end - start + 1
    title = '{}:{}-{}'.format(chrom, start, end)
    if gene:
        title = '{} ({} nt)\n{}'.format(gene, exon_length, title)
    else:
        title = '{} nt microexon ({} strand)\n{}'.format(exon_length, strand, title)
    heat_axes.set_title(title, fontsize=19, pad=14)

    if subtitle:
        figure.text(0.5, 0.03, subtitle, ha='center', fontsize=10, color='#B00020')
    if panel_label:
        figure.text(0.01, 0.93, panel_label, fontsize=26, fontweight='bold')

    return figure


def write_tsv(path, matrix, chrom, start, end, strand, gap_label='microexon'):
    """Dump the delta matrix as a tab separated table."""
    positions = list(range(-FLANK_SIZE, 0)) + list(range(1, FLANK_SIZE + 1))
    with open(path, 'w') as handle:
        handle.write('chr\tstart\tend\tstrand\tposition\tflank\t{}\n'.format(
            '\t'.join(MUTATION_BASES)))
        for index, position in enumerate(positions):
            flank = 'upstream' if position < 0 else 'downstream'
            handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                chrom, start, end, strand, position, flank,
                '\t'.join('{:.6g}'.format(value) for value in matrix[index])))


def demo_matrix(seed=7):
    """Synthetic scores, for checking the layout without model or genome."""
    random = np.random.RandomState(seed)
    positions = np.arange(2 * FLANK_SIZE)
    envelope = 0.004 + 0.008 * np.exp(-((positions - 60) ** 2) / 600.0)
    envelope += 0.006 * np.exp(-((positions - 150) ** 2) / 2000.0)
    return random.normal(0, 1, (2 * FLANK_SIZE, len(MUTATION_BASES))) * envelope[:, None]


def parse_exon(exon):
    fields = exon.split(':')
    if len(fields) != 4:
        raise ValueError(
            "--exon must look like chr1:100020:100030:+, got {!r}".format(exon))
    chrom, start, end, strand = fields
    return chrom, int(start), int(end), strand


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Mutational screening heatmap (PositionScore) of a microexon.')
    parser.add_argument('--exon', required=True,
                        help='Exon coordinate Ex: chrX:31126642:31126673:+')
    parser.add_argument('--model', help='CNN model file (src/saved_model.hdf5)')
    parser.add_argument('--genome', help='HG38 genome fasta file')
    parser.add_argument('--conservation', help='HG38 conservation bigwig file')
    parser.add_argument('--gene', help='Gene name printed in the title')
    parser.add_argument('--output', default='positionscore.png',
                        help='Output figure; the extension picks the format (png, pdf, svg, tif)')
    parser.add_argument('--tsv', help='Also write the delta scores to this tab separated file')
    parser.add_argument('--vmax', type=float, default=0.015,
                        help='Color scale limit, the scale runs from -vmax to +vmax (default: 0.015)')
    parser.add_argument('--gap', type=int, default=4,
                        help='Blank columns drawn between the flanks (default: 4)')
    parser.add_argument('--base-order', default=FIGURE_BASE_ORDER,
                        help='Heatmap row order, top to bottom (default: TGAC)')
    parser.add_argument('--panel-label', help='Panel letter drawn on the top left corner')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Variants scored per model call (default: 256)')
    parser.add_argument('--dpi', type=int, default=300, help='Output resolution (default: 300)')
    parser.add_argument('--demo', action='store_true',
                        help='Render synthetic scores instead of running the model')
    parser.add_argument('--quiet', action='store_true', help='Do not report progress')

    args = parser.parse_args(argv)

    if sorted(args.base_order) != sorted(MUTATION_BASES):
        parser.error('--base-order must be a permutation of ACGT, got {!r}'.format(args.base_order))

    try:
        chrom, start, end, strand = parse_exon(args.exon)
    except ValueError as error:
        parser.error(str(error))
    if strand not in ('+', '-'):
        parser.error("strand must be '+' or '-', got {!r}".format(strand))

    subtitle = None
    if args.demo:
        matrix = demo_matrix()
        wild_type_score = float('nan')
        subtitle = 'DEMO: synthetic scores, not a model prediction'
    else:
        missing = [name for name in ('model', 'genome', 'conservation')
                   if getattr(args, name) is None]
        if missing:
            parser.error('--{} required (or use --demo)'.format(', --'.join(missing)))

        import deepmex

        deepmex.set_reference_files(args.genome, args.conservation)
        if not args.quiet:
            print('loading model {}'.format(args.model), file=sys.stderr)
        model = deepmex.load_cnn_model(args.model)
        microexon = deepmex.Microexon(chrom, start, end, strand)
        matrix, wild_type_score = position_score_matrix(
            microexon, model, batch_size=args.batch_size, verbose=not args.quiet)
        print('Microexon prediction: {}'.format(wild_type_score))

    matrix = to_transcript_orientation(matrix, strand)

    if args.tsv:
        write_tsv(args.tsv, matrix, chrom, start, end, strand)
        print('scores written to {}'.format(args.tsv))

    figure = plot_position_scores(
        matrix, chrom, start, end, strand, gene=args.gene, vmax=args.vmax,
        gap=args.gap, base_order=args.base_order, panel_label=args.panel_label,
        subtitle=subtitle)
    figure.savefig(args.output, dpi=args.dpi)
    print('figure written to {}'.format(args.output))


if __name__ == '__main__':
    main()
