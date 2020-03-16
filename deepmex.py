from keras.models import load_model
import numpy as np
import pyBigWig
import pybedtools
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 




parser = argparse.ArgumentParser()
parser.add_argument("--model", help="CNN model file")
parser.add_argument("--genome", help="HG38 genome file")
parser.add_argument("--conservation", help="conservation bigwig file hg38.100way.phastCons.bw")
parser.add_argument("--exon", help="A exon coordinate Ex: chr1:100020:100030:+")

args = parser.parse_args()


model_file = args.model


model_merged = load_model(model_file)


base_encode = {
    'A': [0, 0, 0, 1],
    'C': [0, 0, 1, 0],
    'T': [0, 1, 0, 0],
    'G': [1, 0, 0, 0]
}


GENOME= args.genome #'/srv/notebooks/microexon_predictor/hg38.fa' 
CONSERVATION_FILE =  args.conservation #'/srv/notebooks/microexon_predictor/hg38.100way.phastCons.bw' 


def capture_values_bw(big_wig, 
                      chr,
                      start,
                      end):
    try:
        with pyBigWig.open(big_wig) as bw:
            values_vector  = bw.values(chr, start, end)
            return values_vector
    except Exception as e:
        print (chr,start,end)
        print (e)
        
        
def capture_fasta_sequences(fasta_file, 
                      chr,
                      start,
                      end):



    fasta_query = pybedtools.BedTool("{} {} {}".format(chr, start, end), from_string=True)
    fasta_query_out = fasta_query.seq((chr, start, end), fasta_file)
    return fasta_query_out



class Microexon():
    def __init__(self, chr, start, end, strand):
        self.chr  = chr
        self.start  = start
        self.end  = end
        self.strand  = strand
        self.flank_up_limit_start = self.start - 1
        self.flank_up_limit_end = self.start - 101
        self.flank_down_limit_start = self.end + 1
        self.flank_down_limit_end = self.end + 101
        self.conservation_values_up = None
        self.conservation_values_down = None
        self.seq_values_up = None
        self.seq_values_down = None
        self.set_values()
        self.encode_dna_up = self.encoding_dna_sequence(self.seq_values_up)
        self.encode_dna_down = self.encoding_dna_sequence(self.seq_values_down) 
        
        

        
    def set_values(self):
        '''

        '''
        self.conservation_values_up = np.array(capture_values_bw(CONSERVATION_FILE,
                          self.chr,
                          self.flank_up_limit_end,
                          self.flank_up_limit_start)).reshape(1,100,1)
        
        self.conservation_values_down = np.array(capture_values_bw(CONSERVATION_FILE,
                  self.chr,
                  self.flank_down_limit_start,
                  self.flank_down_limit_end)).reshape(1,100,1)

        
        self.seq_values_up = capture_fasta_sequences(GENOME,
                          self.chr,
                          self.flank_up_limit_end,
                          self.flank_up_limit_start)
        
        self.seq_values_down = capture_fasta_sequences(GENOME,
                  self.chr,
                  self.flank_down_limit_start,
                  self.flank_down_limit_end)
        
        
#         print (map(len, [self.conservation_values_up,
#                          self.conservation_values_down,
#                         self.seq_values_up,
#                         self.seq_values_down,]))
        
    def encoding_dna_sequence(self, x):
        return np.array([base_encode[x_nucleo.upper()] for x_nucleo in x ])
    
    
    
    def prediction(self):
        '''I need to flip strands at this point'''
        if self.strand == '+':
            out_prob = model_merged.predict([self.encode_dna_up.reshape(1, 100, 4),
                     self.conservation_values_up.reshape(1,100,1),
                     self.encode_dna_down.reshape(1, 100, 4),
                     self.conservation_values_down.reshape(1,100,1)] )
            
            
        if self.strand == '-':
            out_prob = model_merged.predict([self.encode_dna_down[::-1].reshape(1, 100, 4),
                     self.conservation_values_down[::-1].reshape(1, 100, 1),
                     self.encode_dna_up[::-1].reshape(1, 100, 4),
                     self.conservation_values_up[::-1].reshape(1,100,1)] )
        
        print (out_prob[0][0] * 100, "Microexon score")
        
coord = args.exon.split(':')
print (coord)
m1 = Microexon(coord[0], int(coord[1]), int(coord[2]),coord[3])
print ('======================================================')
print ('======================================================')
m1.prediction()        
print ('======================================================')
print ('======================================================')
