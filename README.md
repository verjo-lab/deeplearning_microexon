# DeepMEx: Deeplearning MicroExon
Predicting microexons using a Convolutional Neural Network (CNN) model.

## Requirements

Python 3.13 or newer (the project is pinned to 3.15 in `.python-version`) and
[uv](https://docs.astral.sh/uv/). Make sure `bedtools` is installed and on your
PATH.

```
uv sync                    # demo + tests (Python 3.15)
```

TensorFlow does not publish CPython 3.15 wheels yet, so the environment that
actually runs the CNN is built on Python 3.13:

```
uv sync --extra model --extra genome --python 3.13    # or: make install-model
```

Optional extras: `model` (tensorflow, keras), `genome` (pyBigWig, pybedtools),
`gtf` (pandas, gtfparse, tqdm, for the training set preparation).

Use `make download-data` to download the hg38 fasta and bigwig files (if you
already have the files, move them to `src/data/`).

## deepmex: predicting a microexon

`deepmex` predicts a human microexon given an exon chromosome coordinate.

### Parameters

 - model: the CNN model file (`src/saved_model.hdf5` is the default model designed for this application)
 - genome: the Hg38 fasta file (http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/ or `make download-data`)
 - conservation: the Hg38 conservation BigWig file (http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ or `make download-data`)
 - exon: the exon coordinate to be verified as a microexon, i.e.: chr1:100020:100030:+

### Expected output

OUTPUT: (min value 0, max value 100)

  Microexon score: score > 50 are predicted as microexons.

### Sample command

```
uv run deepmex --model src/saved_model.hdf5 --genome src/data/hg38.fa \
  --conservation src/data/hg38_cons.bw --exon chr1:100020:100030:+ > result.out
```

## PositionScore: mutational screening figure

`positionscore` runs the synthetic mutational screening of the training
notebook from the command line and draws the heatmap used in the paper
(Fig. 7A): every base of the two 100 nt flanks is replaced by each of the four
nucleotides, the microexon is re-scored, and the delta (mutant - wild type) is
plotted as a 4 x 200 heatmap.

### Parameters

 - exon (required): the microexon coordinate, i.e.: chrX:31126642:31126673:-
 - model, genome, conservation: the same files used by `deepmex`
 - gene: gene name printed in the title, i.e.: DMD
 - output: output figure, the extension picks the format (png, pdf, svg, tif; tif is LZW compressed). Default: positionscore.png
 - tsv: also write the per position / per base scores to a tab separated file
 - vmax: color scale limit, the scale runs from -vmax to +vmax. Default: 1.5
   (the published figure uses 0.015 on the 0-1 scale of the notebook; the scores
   reported here are on the 0-100 scale, hence the factor of 100)
 - gap: blank columns drawn between the two flanks. Default: 4
 - base-order: heatmap row order, top to bottom. Default: TGAC
 - panel-label: panel letter drawn on the top left corner, i.e.: A
 - batch-size: variants scored per model call. Default: 256
 - dpi: output resolution. Default: 300
 - demo: render the layout with synthetic scores, without model or reference files

Panel B (inclusion of the microexon under a knock-down) is drawn from a
vast-tools table, not from the model. Give all three and the output becomes the
two panel figure:

 - knockdown: an `INCLUSION_LEVELS_FULL` table
 - event: the vast-tools event id, i.e. HsaEX0019952
 - group: `NAME=SAMPLE,SAMPLE`, passed twice, the treated group first
 - knockdown-title: title of panel B, i.e. "PTBP1 knock-down in HepG2 cells"

The densities are the Beta posteriors of the inclusion level built from the
corrected read counts of the quality column (`...@inclusion,exclusion`), and
the right hand curve is P(|dPSI| > x) with the dashed line at the largest
difference supported at 95%, the `MV[dPsi]` of vast-tools.

```
uv run positionscore --model src/saved_model.hdf5 --genome src/data/hg38.fa \
  --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:- --gene DMD \
  --knockdown INCLUSION_LEVELS_FULL.tab --event HsaEX0019952 \
  --group shRNA=HepG2_sh1,HepG2_sh2 --group Control=HepG2_ctl1,HepG2_ctl2 \
  --knockdown-title "PTBP1 knock-down in HepG2 cells" --output Fig7AB.tif
```

Panel C of the published figure is a UCSC browser screenshot and is not
produced here.

### Sample command

```
uv run positionscore --model src/saved_model.hdf5 --genome src/data/hg38.fa \
  --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:- \
  --gene DMD --panel-label A --output DMD_positionscore.tif --tsv DMD_positionscore.tsv
```

To check the figure layout without downloading the reference files (only numpy
and matplotlib are needed):

```
uv run positionscore --exon chrX:31126642:31126673:- --gene DMD --demo --output demo.png
```

The screening scores 800 variants (200 positions x 4 bases) in batches, so a
single microexon takes one model load plus a few seconds of prediction.

Minus strand exons are drawn in transcript orientation: the position axis is
reversed *and* the bases are complemented, so column 0 is always -100 relative
to the exon and a row labelled `A` is the mutation to A of the transcript.
This reproduces the published panel (DMD is on the minus strand, so pass
`chrX:31126642:31126673:-`).

`--output` picks the format from the extension; `.tif` is written with lossless
LZW compression, which takes a 300 dpi panel from ~20 MB down to ~2 MB.

## Development

```
make test      # uv run pytest
make lint      # ruff check + ruff format --check
make format    # ruff format + ruff check --fix
```

The package lives in `src/deepmex/`:

 - `core.py`: reference file access, one hot encoding and the `Microexon` scoring
 - `cli.py`: the `deepmex` command line
 - `positionscore.py`: the mutational screening, the figure composition and the `positionscore` command line
 - `knockdown.py`: panel B, from a vast-tools table
 - `gtf.py`: exon tables from a GTF annotation (extra `gtf`)
 - `conservation.py`: per base conservation of the flanks, for training set preparation (extra `gtf`)

## Other files available

### src/training_notebooks/model_training_microexons.ipynb

A Jupyter notebook file containing the steps to train the CNN and save the model.

Some steps can be modified to generate new species models. This file also contains the procedure to create the synthetic mutational microexons screening (**PositionScore**).

## Next Releases:

  -Generate a conda package.
