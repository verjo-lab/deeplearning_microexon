# DeepMEx: Deeplearning MicroExon
Predicting microexons using a Convolutional Neural Network (CNN) model.  

 __deepmex.py__ :  
A command line application to predict human microexons given an exon chromosome coordinate.  


```
PARAMETERS


--model  "CNN model file"  ( This repository contains a hdf5 file with the model used in our paper)
--genome "HG38 genome file"  (download at http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/)
--conservation "conservation bigwig file hg38.100way.phastCons.bw" (download at http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/)  
--exon "A exon coordinate Ex: chr1:100020:100030:+"  
  

  
OUTPUT: (min value 0, max value 100)  

  Microexon score: score > 50 are predicted  as microexons.  

```


```
Usage:  

python human_microexon_predictor.py --model model.hdf5 --genome hg38.fa --conservation 
hg38.100way.phastCons.bw --exon chr1:100020:100030:+ > result.out

```
__model_training_microexons.ipynb__:  
  

A Jupyter notebook file containing the steps to train the CNN and save the model. 
Some steps can be modified to generate new species models.  
This file also contains the procedure to create the synthetic mutational microexons screening (**PositionScore**).  
 





__Next Releases:__  

  -Generate a conda package.  

  -Include the mutational screening in the command line tool.



