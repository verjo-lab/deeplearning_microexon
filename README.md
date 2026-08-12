# DeepMEx: Deeplearning MicroExon
Predicting microexons using a Convolutional Neural Network (CNN) model.  


## How to run:

```
WARNING: This is a tool that is designed to run on Python 2.7.X. Please be aware that running this tool on Python 3.X versions may result on error that are not solved yet.
```

Make sure you have bedtools installed in your environment and added to your PATH env variable.

Use the `make download-data` to download the hg38 Fasta and BigWig files (if you already have the files, move them to the path: `src/data/`).

To get the prediction, use `deepmex.py` command line, which is an application to predict human microexons given an exon chromosome coordinate.

### Parameters
Deepmex receives multiple arguments as a requirement to be used. All of them are documented below:

 - model: the CNN model file (src/saved_model.hdf5 is the default model designed for this application)
 - genome: the Hg38 Fasta File (downloadable through http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/ or `make download-data`)
 - conservation: the Hg38 conservation BigWig file (downloadable through http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ or `make download-data`)  
 - exon: The exon coordinate to be verified as a microexon, i.e.: chr1:100020:100030:+

### Expected output:

OUTPUT: (min value 0, max value 100)  

  Microexon score: score > 50 are predicted  as microexons.  


### Sample command

```
python human_microexon_predictor.py --model model.hdf5 --genome hg38.fa --conservation 
hg38.100way.phastCons.bw --exon chr1:100020:100030:+ > result.out
```

## PositionScore: mutational screening figure

`positionscore.py` runs the synthetic mutational screening of the training
notebook from the command line and draws the heatmap used in the paper
(Fig. 7A): every base of the two 100 nt flanks is replaced by each of the four
nucleotides, the microexon is re-scored, and the delta (mutant - wild type) is
plotted as a 4 x 200 heatmap.

### Parameters

 - exon (required): the microexon coordinate, i.e.: chrX:31126642:31126673:+
 - model, genome, conservation: the same files used by `deepmex.py`
 - gene: gene name printed in the title, i.e.: DMD
 - output: output figure, the extension picks the format (png, pdf, svg, tif). Default: positionscore.png
 - tsv: also write the per position / per base scores to a tab separated file
 - vmax: color scale limit, the scale runs from -vmax to +vmax. Default: 0.015
 - gap: blank columns drawn between the two flanks. Default: 4
 - base-order: heatmap row order, top to bottom. Default: TGAC
 - panel-label: panel letter drawn on the top left corner, i.e.: A
 - batch-size: variants scored per model call. Default: 256
 - dpi: output resolution. Default: 300
 - demo: render the layout with synthetic scores, without model or reference files

### Sample command

```
python src/positionscore.py --model src/saved_model.hdf5 --genome src/data/hg38.fa \
  --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:+ \
  --gene DMD --panel-label A --output DMD_positionscore.png --tsv DMD_positionscore.tsv
```

To check the figure layout without downloading the reference files (only numpy
and matplotlib are needed):

```
python src/positionscore.py --exon chrX:31126642:31126673:+ --gene DMD --demo --output demo.png
```

The screening scores 800 variants (200 positions x 4 bases) in batches, so a
single microexon takes one model load plus a few seconds of prediction.

Minus strand exons are drawn in transcript orientation (column 0 is always
-100 relative to the exon), following the same flank handling `deepmex.py`
applies when it scores a minus strand microexon.

## Other files available

### src/training_notebooks/model_training_microexons.ipynb:  

A Jupyter notebook file containing the steps to train the CNN and save the model.

Some steps can be modified to generate new species models. This file also contains the procedure to create the synthetic mutational microexons screening (**PositionScore**).  


## Next Releases:

  -Generate a conda package.  