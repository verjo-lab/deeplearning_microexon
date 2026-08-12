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

 - exon (required): the microexon coordinate, i.e.: chrX:31126642:31126673:+
 - model, genome, conservation: the same files used by `deepmex`
 - gene: gene name printed in the title, i.e.: DMD
 - output: output figure, the extension picks the format (png, pdf, svg, tif). Default: positionscore.png
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

### Sample command

```
uv run positionscore --model src/saved_model.hdf5 --genome src/data/hg38.fa \
  --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:+ \
  --gene DMD --panel-label A --output DMD_positionscore.png --tsv DMD_positionscore.tsv
```

To check the figure layout without downloading the reference files (only numpy
and matplotlib are needed):

```
uv run positionscore --exon chrX:31126642:31126673:+ --gene DMD --demo --output demo.png
```

The screening scores 800 variants (200 positions x 4 bases) in batches, so a
single microexon takes one model load plus a few seconds of prediction.

Minus strand exons are drawn in transcript orientation (column 0 is always
-100 relative to the exon), following the same flank handling `deepmex`
applies when it scores a minus strand microexon.

## Development

```
make test      # uv run pytest
make lint      # ruff check + ruff format --check
make format    # ruff format + ruff check --fix
```

The package lives in `src/deepmex/`:

 - `core.py`: reference file access, one hot encoding and the `Microexon` scoring
 - `cli.py`: the `deepmex` command line
 - `positionscore.py`: the mutational screening and the `positionscore` command line
 - `gtf.py`: exon tables from a GTF annotation (extra `gtf`)
 - `conservation.py`: per base conservation of the flanks, for training set preparation (extra `gtf`)

## Other files available

### src/training_notebooks/model_training_microexons.ipynb

A Jupyter notebook file containing the steps to train the CNN and save the model.

Some steps can be modified to generate new species models. This file also contains the procedure to create the synthetic mutational microexons screening (**PositionScore**).

## Next Releases:

  -Generate a conda package.
