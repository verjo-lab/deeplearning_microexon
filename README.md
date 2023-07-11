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

## Other files available

### src/training_notebooks/model_training_microexons.ipynb:  

A Jupyter notebook file containing the steps to train the CNN and save the model.

Some steps can be modified to generate new species models. This file also contains the procedure to create the synthetic mutational microexons screening (**PositionScore**).  


## Next Releases:

  -Generate a conda package.  

  -Include the mutational screening in the command line tool.