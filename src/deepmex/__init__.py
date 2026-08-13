"""DeepMEx: predicting human microexons with a convolutional neural network."""

from deepmex.core import (
    BASE_ENCODING,
    FLANK_SIZE,
    CNNModel,
    Microexon,
    ReferenceFiles,
    Strand,
    encode_sequence,
    flank_intervals,
    load_cnn_model,
    parse_coordinate,
    parse_exon,
    read_conservation,
    read_sequence,
)

__version__ = "0.2.0"

__all__ = [
    "BASE_ENCODING",
    "FLANK_SIZE",
    "CNNModel",
    "Microexon",
    "ReferenceFiles",
    "Strand",
    "__version__",
    "encode_sequence",
    "flank_intervals",
    "load_cnn_model",
    "parse_coordinate",
    "parse_exon",
    "read_conservation",
    "read_sequence",
]
