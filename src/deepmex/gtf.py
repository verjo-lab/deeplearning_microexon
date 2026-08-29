"""Helpers to slice a GTF annotation into the exon tables used by DeepMEx.

Requires the ``gtf`` extra (``uv sync --extra gtf``).
"""

from pathlib import Path

import pandas as pd

BED6_COLUMNS = ("chr", "start", "end", "strand")


class GTFProcessing:
    """A GTF annotation loaded as a pandas dataframe, with exon level views."""

    def __init__(self, gtf_file_name: str | Path) -> None:
        self.gtf_file_name = Path(gtf_file_name)
        self.df_gtf = self._load_gtf()
        self.df_gtf = self.df_gtf.rename(columns={"seqname": "chr"})
        self.df_gtf["interval_length"] = self.df_gtf["end"] - self.df_gtf["start"]

    def _load_gtf(self) -> pd.DataFrame:
        import gtfparse  # noqa: PLC0415  (optional dependency, see the gtf extra)

        print(f"loading {self.gtf_file_name}...")
        return gtfparse.read_gtf(self.gtf_file_name, result_type="pandas")

    def get_gtf_df(self) -> pd.DataFrame:
        """Return the annotation as a pandas dataframe."""
        return self.df_gtf

    @staticmethod
    def remove_dup_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop repeated column names, keeping the first occurrence."""
        return frame.loc[:, ~frame.columns.duplicated()]

    @staticmethod
    def get_first_exon_df(gtf_df: pd.DataFrame) -> pd.DataFrame:
        """Return the first exon of every transcript, relative to the strand."""
        exons = gtf_df[gtf_df["feature"] == "exon"]
        first = exons[exons["exon_number"].astype(str) == "1"]
        return first.sort_values("transcript_id")

    @staticmethod
    def get_last_exon_df(gtf_df: pd.DataFrame) -> pd.DataFrame:
        """Return the last exon of every transcript, relative to the strand."""
        exons = gtf_df[gtf_df["feature"] == "exon"]
        return GTFProcessing._extreme_exon_per(exons, "transcript_id")

    @staticmethod
    def _extreme_exon_per(exons: pd.DataFrame, key: str) -> pd.DataFrame:
        """Rightmost exon of plus strand groups, leftmost of minus strand ones."""
        plus = exons[exons["strand"] == "+"].sort_values("end").groupby(key, sort=False).tail(1)
        minus = exons[exons["strand"] == "-"].sort_values("start").groupby(key, sort=False).head(1)
        return pd.concat([plus, minus]).sort_values(key)

    @staticmethod
    def capture_distal_unique_tes(gtf_df: pd.DataFrame) -> pd.DataFrame:
        """Return the most distal transcription end site of every gene."""
        last_exons = GTFProcessing.get_last_exon_df(gtf_df)
        return GTFProcessing._extreme_exon_per(last_exons, "gene_id")

    @staticmethod
    def capture_distal_unique_tss(gtf_df: pd.DataFrame) -> pd.DataFrame:
        """Return the most distal transcription start site of every gene."""
        first_exons = GTFProcessing.get_first_exon_df(gtf_df)
        plus = first_exons[first_exons["strand"] == "+"]
        minus = first_exons[first_exons["strand"] == "-"]
        distal = pd.concat(
            [
                plus.sort_values("start").groupby("gene_id", sort=False).head(1),
                minus.sort_values("end").groupby("gene_id", sort=False).tail(1),
            ]
        )
        return distal.sort_values("gene_id")

    @staticmethod
    def df_to_df_bed(
        gtf_df: pd.DataFrame,
        fourth_position_feature: str = "gene_name",
        fifth_position_feature: str = "transcript_id",
    ) -> pd.DataFrame:
        """Return the annotation in bed6 column order."""
        columns = ["chr", "start", "end", fourth_position_feature, fifth_position_feature, "strand"]
        return gtf_df[columns]

    @staticmethod
    def df_to_bed(
        gtf_df: pd.DataFrame,
        bed_file_name: str | Path,
        fourth_position_feature: str = "gene_name",
        fifth_position_feature: str = "transcript_id",
    ) -> Path:
        """Write the annotation as a bed6 file and return its path."""
        bed = GTFProcessing.df_to_df_bed(gtf_df, fourth_position_feature, fifth_position_feature)
        bed.to_csv(bed_file_name, sep="\t", header=False, index=False)
        return Path(bed_file_name)

    @staticmethod
    def hist_generate(gtf_df: pd.DataFrame, feature: str = "transcript_biotype") -> None:
        """Bar plot of the first exons grouped by ``feature``.

        Example: ``GTFProcessing.hist_generate(gtf_df.head(1600), 'transcript_biotype')``
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        counts = GTFProcessing.get_first_exon_df(gtf_df).groupby(feature).size()
        plt.bar(range(counts.size), counts.to_numpy())
        plt.xticks(range(counts.size), counts.index.to_numpy(), rotation="vertical")
        plt.title(feature)
        plt.show()

    @staticmethod
    def generate_hist_by_transcript_biotypes(gtf_df: pd.DataFrame) -> None:
        GTFProcessing.hist_generate(gtf_df, feature="transcript_biotype")
