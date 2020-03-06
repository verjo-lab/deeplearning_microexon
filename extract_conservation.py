__author__ = 'lucas'
# -*- coding: utf-8 -*-
import sys
import pandas as pd
from tqdm import tqdm
from pybedtools import BedTool
import os



def get_conservation_hash(size_sequence, flanks_regions):
    creating_flank_regions = []
    for f_i in flanks_regions[['chr', 'start', 'end', 'short_id']].values:

        for p in ['up', 'down']:
            f = f_i[:]  # copiando objeto
            indice = map(int, f[-1].split('_')[1:])
            if p =='up':

                start_tres_linha = (indice[0]-1)

                start_cinco_linha = start_tres_linha - size_sequence

                f[1] = start_cinco_linha
                f[2] = start_tres_linha

                creating_flank_regions.append('chr' + '\t'.join(map(str, f))+'_'+p)

            else:

                start_cinco_linha = (indice[1] + 1)
                start_tres_linha = start_cinco_linha + size_sequence

                f[1] = start_cinco_linha
                f[2] = start_tres_linha


                creating_flank_regions.append('chr' + '\t'.join(map(str, f)) + '_' + p)



    flank_list_str = '\n'.join(creating_flank_regions)
    print flank_list_str



    print 'sorting...flank file'
    flank_bedtools = BedTool(flank_list_str, from_string=True).sort()
    flank_bedtools.saveas('/home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flank.bed')

    print 'longo...intersect....'
    cmd = 'bedmap --skip-unmapped --echo  --echo-map /work/users/vinicius/xto/hg38.bed.graph  /home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flank.bed > /home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flanks_conser_bedtools.bed'
    os.system(cmd)

    # ['chr1',
    #  '940918',
    #  '940922',  <========== essa ponta com um bloco esta passando e entrando  do exon (colocar limite)
    #  '0',
    #  'chr1',
    #  '940920',
    #  '941170',

    print 'finished ...intersect....'
    conser_bed = '/home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flanks_conser_bedtools.bed'
    separating_multiple_hits=[]
    file_creating_out_fixed_multiple = '/home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flanks_conser_bedtools_fixed.bed'
    creating_out_fixed_multiple = open(file_creating_out_fixed_multiple, 'w')
    conser_to_sep = open(conser_bed).read().split('\n')
    print 'creating_out_fixed_multiple...'
    for l_c in tqdm(conser_to_sep):
        if ";" in l_c:
            l_c_splited = l_c.replace('|', ';').split(';')
            for insert_line in [l_c_splited[0] + '\t' +  l_c_multiple for l_c_multiple in l_c_splited[1:]]:
                separating_multiple_hits.append(insert_line)
        else:
            separating_multiple_hits.append(l_c.replace('|', '\t'))

    creating_out_fixed_multiple.write('\n'.join(separating_multiple_hits)) #  PROBLEMA NA RAM AQUI usar bedpos?
    creating_out_fixed_multiple.close()
    file_conser = pd.read_table(file_creating_out_fixed_multiple, sep='\t', header=None)
    print file_conser

    #Preapring to bedtools intersect

    bedtools_preparing_list = []
    for line in tqdm(file_conser.values[:-1], desc='Preparing to final intersect...'):
        if len(line) == 8:
            end = int(line[2])
            start = int(line[1])
            size = end - start
            conser = float(line[3])
            key_exon = line[-1].strip('_up|_down')
            # print key_exon
            if '_up' in line[-1]:
                out_temp_unpacked = [
                                        '\t'.join(
                                                [line[0],
                                                 str(line[1] + n_con),
                                                 str(line[1] + n_con + 1),
                                                 str(round(conser, ndigits=1))
                                                 ]) for n_con in range(size)

                                             ]
                #print len (out_temp_unpacked)
                bedtools_preparing_list.append('\n'.join(out_temp_unpacked))
                 # adicionando e desempacotando
            else:
                out_temp_unpacked = [
                    '\t'.join(
                            [line[0],
                             str(line[1] + n_con),
                             str(line[1] + n_con + 1),
                             str(round(conser, ndigits=1))
                             ]) for n_con in range(size)
                    ]
                #print len(out_temp_unpacked)
                bedtools_preparing_list.append('\n'.join(out_temp_unpacked))
                # adicionando e desempacotando


    print 'tamanho bedtools_preparing_list: ', len(bedtools_preparing_list)
    print bedtools_preparing_list[0:2]

    map_unpacked = BedTool('\n'.join([o_line for b_line in bedtools_preparing_list for o_line in b_line.split('\n')]), from_string=True).sort()
    map_unpacked.saveas('/work/users/lucassilva/temp/map_unpacked.bed')
    intersected_map = map_unpacked.intersect('/home/lucas/PycharmProjects/MEGS_introns/fixando_consevation/flank.bed', wb=True, wa=True)
    print 'len intersected..', len(intersected_map)

    map_unpacked_intersected = '/work/users/lucassilva/temp/final_flanks_conser_bedtools_fixed.bed'
    intersected_map.saveas(map_unpacked_intersected)
    print 'removing_duplicates....'
    map_unpacked_file_conser = pd.read_table(map_unpacked_intersected, sep='\t', header=None).drop_duplicates()

    hash_conser = {}

    for line in tqdm(map_unpacked_file_conser[:-1].values):
        if len(line)==8:
            key_exon = line[-1].strip('_up|_down')
            hash_conser.setdefault(key_exon, []).append(line) #adicionando e desempacotando

    hash_conser_out = {}




    for k, v in tqdm(hash_conser.iteritems()):
        hash_conser_out[k] = [
            [u[3] for u in v[:size_sequence]],
            [d[3] for d in v[size_sequence:]]

                            ] # erro
        # if len(v) == size_sequence*2:
        #     print 'diferente do tamanho desehado!'
        #     print v[size_sequence:]
        #     print k
        #     print len(v)
        #     print len(v[:size_sequence])
        #     print len(v[size_sequence:])



    return hash_conser_out

def main():
    pass

    #[exit() for check_size in hash_conser.values() if len(check_size)  != 200]
        # if len(hash_conser) >30:
        #     print hash_conser
        #     print [len(v) for v in  hash_conser.values()]
        #     exit()

if __name__ == '__main__':
    sys.exit(main())
