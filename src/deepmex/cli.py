"""Command line interface: predict a human microexon from its coordinate."""

import argparse
import shutil
import sys
from collections.abc import Sequence

from deepmex.core import (
    Microexon,
    ModelUnavailableError,
    ReferenceFiles,
    load_cnn_model,
    parse_exon,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepmex",
        description="Predict a human microexon from a genomic coordinate.",
    )
    parser.add_argument("--model", required=True, help="CNN model file (src/saved_model.hdf5)")
    parser.add_argument("--genome", required=True, help="HG38 genome fasta file")
    parser.add_argument(
        "--conservation",
        required=True,
        help="conservation bigwig file (hg38.100way.phastCons.bw)",
    )
    parser.add_argument("--exon", required=True, help="exon coordinate, i.e. chr1:100020:100030:+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        chrom, start, end, strand = parse_exon(args.exon)
        references = ReferenceFiles.from_paths(args.genome, args.conservation)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    try:
        model = load_cnn_model(args.model)
    except ModelUnavailableError as error:
        print(error, file=sys.stderr)
        return 1
    microexon = Microexon.from_reference(chrom, start, end, strand, references)
    score = microexon.predict(model)

    print("=" * shutil.get_terminal_size().columns)
    print(f"Microexon prediction for {microexon.coordinate}: {score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
