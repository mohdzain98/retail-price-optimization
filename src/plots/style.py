"""Shared plot style: validated categorical palette, neutral inks and rcParams.

Every chart draws colours and inks from PlotStyle so figures read as one system;
categorical slots follow the entity (category, model), never its rank.
FigureStore saves figures to the reports folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotStyle:
    """Palette (categorical slots in fixed order, never cycled) and neutral inks.

    `apply()` writes the matching matplotlib rcParams.
    """

    series: tuple[str, ...] = (
        "#257de0",
        "#f47230",
        "#00b894",
        "#f2a600",
        "#ee7fac",
        "#008a17",
        "#6a4ce2",
        "#ec4647",
    )
    ink: str = "#191919"
    ink_secondary: str = "#666666"
    grid: str = "#e6e6e6"
    sequential_cmap: str = "Blues"

    def apply(self) -> None:
        """Set global rcParams for every subsequent figure."""
        plt.rcParams.update(
            {
                "figure.dpi": 110,
                "savefig.dpi": 150,
                "savefig.bbox": "tight",
                "axes.edgecolor": self.grid,
                "axes.labelcolor": self.ink_secondary,
                "axes.titlecolor": self.ink,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.grid": True,
                "axes.axisbelow": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.color": self.grid,
                "grid.linewidth": 0.6,
                "xtick.color": self.ink_secondary,
                "ytick.color": self.ink_secondary,
                "legend.frameon": False,
                "legend.labelcolor": self.ink_secondary,
                "lines.linewidth": 2,
                "font.size": 9,
            }
        )

    def ordered_colors(self, keys: list[str]) -> dict[str, str]:
        """Slot per key in the given (fixed) order; callers pass a stable order."""
        return {k: self.series[i % len(self.series)] for i, k in enumerate(keys)}

    def category_colors(self, categories: list[str]) -> dict[str, str]:
        """Slot per category by sorted name, so colour follows the entity."""
        return self.ordered_colors(sorted(categories))


class FigureStore:
    """Writes figures as PNG files into one folder (reports/figures by default).

    Plotters only draw; saving is this class's single job.
    """

    def __init__(self, figures_dir: Path) -> None:
        self._figures_dir = figures_dir

    def save(self, fig: Figure, name: str) -> Path:
        """Save fig as <figures_dir>/<name>.png and return the path."""
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        path = self._figures_dir / f"{name}.png"
        fig.savefig(path)
        return path
