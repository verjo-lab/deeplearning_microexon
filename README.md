# deeplearning_microexon
Predicting microexons using a Convolutional Neural Network (CNN) model.  

 __human_microexon_predictor.py__ :  
A command line application to predict human microexons given an exons coordinate.  
  
--model  "CNN model file"  
--genome "HG19 genome file"  
--conservation "conservation big wig file hg19.100way.phastCons.bw"  
--exon "A exon coordinate Ex: chr1:100020:100030:+"  



__model_training_microexons.ipynb__:  
A Jupiter notebook file containing the steps to train the CNN and save the model. Some steps can be modified to generate new species models.

