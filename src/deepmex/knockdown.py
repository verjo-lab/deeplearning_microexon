"""Panel B: inclusion of a microexon under a knock-down, from vast-tools.

``vast-tools`` reports, per sample, the point estimate of the inclusion level
(PSI) and the corrected number of reads supporting inclusion and exclusion.
Those counts give a Beta posterior per replicate, which is what the published
panel draws: the posterior of each group on the left, and the probability that
the difference between the two groups exceeds a given value on the right.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Colors of the published panel.
GROUP_COLORS = {"shRNA": "#E3B33F", "Control": "#A9D0E8"}
FALLBACK_COLORS = ["#E3B33F", "#A9D0E8", "#B4C97E", "#D79BC0"]
THRESHOLD_COLOR = "#CC3311"

#: A panel compares a treated group with a control, and the read counts of a
#: vast-tools quality score are the inclusion/exclusion pair.
PAIR = 2

#: Draws taken from each replicate posterior when combining a group.
DRAWS = 20_000

#: Confidence used to report the largest difference still supported, the
#: ``MV[dPsi]`` of vast-tools.
CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class Replicate:
    """One sample of one group."""

    sample: str
    psi: float
    inclusion: float
    exclusion: float

    def posterior(self, draws: int, rng: np.random.Generator) -> np.ndarray:
        """Beta posterior of the inclusion level, with a uniform prior."""
        return rng.beta(self.inclusion + 1, self.exclusion + 1, draws)


@dataclass(frozen=True, slots=True)
class KnockdownEvent:
    """The replicates of every group for a single splicing event."""

    event_id: str
    gene: str
    coordinate: str
    groups: dict[str, list[Replicate]]

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    def group_posterior(self, group: str, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
        """Posterior of the group, averaging one draw per replicate."""
        rng = np.random.default_rng(seed)
        replicates = self.groups[group]
        if not replicates:
            raise ValueError(f"group {group!r} has no replicates")
        return np.mean([r.posterior(draws, rng) for r in replicates], axis=0)

    def difference(self, first: str, second: str, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
        """Posterior of ``PSI(first) - PSI(second)``."""
        return self.group_posterior(first, draws, seed) - self.group_posterior(second, draws, seed + 1)


def parse_quality(quality: str) -> tuple[float, float]:
    """Read the corrected inclusion/exclusion counts of a vast-tools score.

    The quality column ends with ``@inclusion,exclusion``, as in
    ``SOK,SOK,SOK,OK,S@23.00,45.00``.
    """
    if "@" not in quality:
        raise ValueError(f"no read counts in quality score {quality!r}")
    counts = quality.rsplit("@", 1)[1].split(",")
    if len(counts) < PAIR:
        raise ValueError(f"malformed read counts in quality score {quality!r}")
    try:
        return float(counts[0]), float(counts[1])
    except ValueError as error:
        raise ValueError(f"non numeric read counts in quality score {quality!r}") from error


def read_vast_tools(
    path: str | Path,
    event_id: str,
    groups: dict[str, list[str]],
) -> KnockdownEvent:
    """Read one event of an ``INCLUSION_LEVELS_FULL`` table.

    ``groups`` maps a group name to the sample columns that belong to it, as
    in ``{"shRNA": ["HepG2_sh1", "HepG2_sh2"], "Control": [...]}``.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = lines[0].split("\t")
    column = {name: index for index, name in enumerate(header)}

    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) > 1 and fields[1] == event_id:
            break
    else:
        raise ValueError(f"event {event_id!r} not found in {path}")

    parsed: dict[str, list[Replicate]] = {}
    for group, samples in groups.items():
        replicates = []
        for sample in samples:
            if sample not in column:
                raise ValueError(f"sample {sample!r} is not a column of {path}")
            quality_column = f"{sample}-Q"
            if quality_column not in column:
                raise ValueError(f"{quality_column!r} is not a column of {path}")
            inclusion, exclusion = parse_quality(fields[column[quality_column]])
            replicates.append(
                Replicate(
                    sample=sample,
                    psi=float(fields[column[sample]]) / 100.0,
                    inclusion=inclusion,
                    exclusion=exclusion,
                )
            )
        parsed[group] = replicates

    return KnockdownEvent(
        event_id=event_id,
        gene=fields[column.get("GENE", 0)],
        coordinate=fields[column["COORD"]] if "COORD" in column else "",
        groups=parsed,
    )


def maximum_difference(difference: np.ndarray, confidence: float = CONFIDENCE) -> float:
    """Largest ``x`` with ``P(|difference| > x) >= confidence`` (vast-tools MV)."""
    return float(np.quantile(np.abs(difference), 1 - confidence))


def draw_knockdown(
    figure: "Figure",
    event: KnockdownEvent,
    band: tuple[float, float] = (0.0, 1.0),
    title: str | None = None,
    panel_label: str | None = None,
    subtitle: str | None = None,
) -> None:
    """Draw panel B into ``figure``, inside the vertical ``band`` (y0, height)."""
    y0, height = band

    def place(x: float, y: float, width: float, tall: float) -> tuple[float, float, float, float]:
        return (x, y0 + y * height, width, tall * height)

    groups = event.group_names
    if len(groups) != PAIR:
        raise ValueError(f"panel B compares exactly two groups, got {groups}")
    treated, control = groups

    density_axes = figure.add_axes(place(0.10, 0.18, 0.36, 0.52))
    curve_axes = figure.add_axes(place(0.62, 0.18, 0.28, 0.52))

    edges = np.linspace(0, 1, 61)
    for index, group in enumerate(groups):
        color = GROUP_COLORS.get(group, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        density_axes.hist(
            event.group_posterior(group),
            bins=edges,
            density=True,
            color=color,
            alpha=0.75,
            label=group,
            zorder=2,
        )
        # Point estimate of every replicate, below the axis.
        for replicate in event.groups[group]:
            density_axes.plot(replicate.psi, -0.35, "o", color=color, markersize=6, clip_on=False, zorder=3)

    density_axes.set_xlim(-0.02, 1.02)
    density_axes.set_xlabel(r"$\hat{\Psi}$", fontsize=15)
    density_axes.set_ylabel("density", fontsize=12)
    density_axes.tick_params(labelsize=10)
    density_axes.grid(color="#DDDDDD", linewidth=0.7, zorder=0)
    density_axes.set_axisbelow(True)
    for spine in density_axes.spines.values():
        spine.set_color("#444444")
    legend = density_axes.legend(
        title="Samples",
        loc="center left",
        bbox_to_anchor=(1.02, 0.6),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )
    legend._legend_box.align = "left"

    difference = event.difference(treated, control)
    x = np.linspace(0, 1, 201)
    probability = [(np.abs(difference) > value).mean() for value in x]
    threshold = maximum_difference(difference)

    curve_axes.plot(x, probability, color="black", linewidth=1.6, zorder=2)
    curve_axes.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.6, zorder=3)
    curve_axes.text(threshold + 0.02, 0.06, f"{threshold:.2f}", color=THRESHOLD_COLOR, fontsize=13, zorder=3)
    curve_axes.set_xlim(0, 1)
    curve_axes.set_ylim(-0.03, 1.03)
    curve_axes.set_xlabel("x", fontsize=13)
    curve_axes.set_ylabel(r"$P((\hat{\Psi}_1 - \hat{\Psi}_2) > x)$", fontsize=12)
    curve_axes.tick_params(labelsize=10)
    curve_axes.grid(color="#DDDDDD", linewidth=0.7, zorder=0)
    curve_axes.set_axisbelow(True)
    for spine in curve_axes.spines.values():
        spine.set_color("#444444")

    if title:
        figure.text(0.5, y0 + 0.93 * height, title, ha="center", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        y0 + 0.85 * height,
        f"Gene: {event.gene}  Event: {event.event_id}",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    if event.coordinate:
        figure.text(0.5, y0 + 0.76 * height, f"Coordinates: {event.coordinate}", ha="center", fontsize=12)
    if subtitle:
        figure.text(0.5, y0 + 0.03 * height, subtitle, ha="center", fontsize=10, color="#B00020")
    if panel_label:
        figure.text(0.01, y0 + 0.90 * height, panel_label, fontsize=26, fontweight="bold")
