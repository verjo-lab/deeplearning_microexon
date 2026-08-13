.PHONY: install install-model download-data run-example run-positionscore-example positionscore-demo test lint format

# Demo + tests environment (Python 3.15, see .python-version).
install:
	uv sync

# Inference environment: TensorFlow has no CPython 3.15 wheels yet.
install-model:
	uv sync --extra model --extra genome --python 3.13

download-data:
	curl -o src/data/hg38.phastCons100way.bw http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38_cons.bw
	curl -o src/data/hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

run-example:
	uv run deepmex --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chr1:100020:100030:+

run-positionscore-example:
	uv run positionscore --model src/saved_model.hdf5 --genome src/data/hg38.fa --conservation src/data/hg38_cons.bw --exon chrX:31126642:31126673:- --gene DMD --panel-label A --output DMD_positionscore.tif

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
