.PHONY: install install-local-reference install-3.15 download-data run-example run-positionscore-example figure-example positionscore-demo test lint format

# Everything, including the model (Python 3.13, see .python-version).
install:
	uv sync

# Reading the flanks from local files instead of the UCSC API needs bedtools.
install-local-reference:
	uv sync --extra genome

# TensorFlow has no CPython 3.15 wheels yet, so the model does not run there;
# the demo, the tests and the linter do.
install-3.15:
	uv sync --python 3.15

download-data:
	curl -o src/data/hg38.phastCons100way.bw http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38_cons.bw
	curl -o src/data/hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

run-example:
	uv run deepmex --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chr1:100020:100030:+

run-positionscore-example:
	uv run positionscore --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:- --gene DMD --panel-label A --output DMD_positionscore.tif

figure-example:
	uv run deepmex-figure chrX:31126642-31126673

positionscore-demo:
	uv run positionscore --exon chrX:31126642:31126673:- --gene DMD --demo --panel-label A --output demo_positionscore.png

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
