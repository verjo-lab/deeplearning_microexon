# Deeplearning microexon
Predicting microexons using a Convolutional Neural Network (CNN) model.  

 __human_microexon_predictor.py__ :  
A command line application to predict human microexons given an exon chromosome coordinate.  
 
__PARAMETERS__


__--model__  "CNN model file"  ( This repository contains a hdf5 file with the model used in our paper)
__--genome__ "HG19 genome file"  
__--conservation__ "conservation bigwig file hg19.100way.phastCons.bw"  
__--exon__ "A exon coordinate Ex: chr1:100020:100030:+"  
  
  
__OUTPUT__: (min value 0, max value 100)  

  Microexon score: score > 50 are predicted  as microexons.  




__model_training_microexons.ipynb__:  
  

A Jupyter notebook file containing the steps to train the CNN and save the model. 
Some steps can be modified to generate new species models.  
This file also contains the procedure to create the synthetic mutational microexons screening.  
 





__Next Releases:__  

  -Include the mutational screening in the command line tool.



