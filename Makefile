download-data:
	curl -o src/data/hg38.phastCons100way.bw http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38_cons.bw
	curl -o src/data/hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

run-example:
	python src/deepmex.py --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chr1:100020:100030:+

run-positionscore-example:
	python src/positionscore.py --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:+ --gene DMD --panel-label A --output DMD_positionscore.png

positionscore-demo:
	python src/positionscore.py --exon chrX:31126642:31126673:+ --gene DMD --demo --panel-label A --output demo_positionscore.png

test:
	python tests/test_positionscore.py
