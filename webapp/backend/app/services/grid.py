"""Groups completed experiments into the V2 topology x severity grid.

Inside `app.services`, this module answers "which real completed run
corresponds to each of the 3x3 (topology, severity) grid cells CLAUDE.md's
V2.8.1 defines" by matching each completed experiment's `experiment_id`
against a small static naming table -- the grid is a fixed, known set of
nine cells, not something inferred generically from arbitrary output
directories. It reuses `app.services.gallery_reader.list_experiments` for
the underlying file reads and does no I/O of its own. Calibration/smoke runs
(e.g. `compact_medium_comparison_calibration`) never match a cell, since
their `experiment_id` isn't a key in the table below -- only a real,
full-replication grid run ever fills a cell.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from app.services import gallery_reader

TOPOLOGIES = ["Compact", "Standard", "Extended"]
SEVERITIES = ["Light", "Medium", "Heavy"]


class GridCellKey(NamedTuple):
    topology: str
    severity: str


# The nine real (topology, severity) combinations, named exactly as
# CLAUDE.md's V2.6/V2.8.1 name their experiment config files (and therefore
# their `experiment_id`s).
_EXPERIMENT_ID_TO_CELL: dict[str, GridCellKey] = {
    "baseline_port_closure_comparison": GridCellKey("Standard", "Medium"),
    "light_port_disruption_comparison": GridCellKey("Standard", "Light"),
    "heavy_port_disruption_comparison": GridCellKey("Standard", "Heavy"),
    "compact_light_comparison": GridCellKey("Compact", "Light"),
    "compact_medium_comparison": GridCellKey("Compact", "Medium"),
    "compact_heavy_comparison": GridCellKey("Compact", "Heavy"),
    "extended_light_comparison": GridCellKey("Extended", "Light"),
    "extended_medium_comparison": GridCellKey("Extended", "Medium"),
    "extended_heavy_comparison": GridCellKey("Extended", "Heavy"),
}


def build_grid() -> dict[str, Any]:
    """Returns the 3x3 grid: one cell per (topology, severity), each either
    the most recent matching completed experiment or `None` if that cell
    hasn't been run yet -- a missing cell is a labeled blank, never omitted
    or treated as zero (mirroring `analysis/plot_results.py`'s existing
    convention for an incomplete grid). `list_experiments()` is already
    sorted newest-first, so the first match found for a cell is its most
    recent real run.
    """
    best: dict[GridCellKey, dict[str, Any]] = {}
    for item in gallery_reader.list_experiments():
        experiment_id = item["manifest"].get("experiment_id", "")
        cell = _EXPERIMENT_ID_TO_CELL.get(experiment_id)
        if cell is None or cell in best:
            continue
        best[cell] = item

    cells: list[dict[str, Any]] = []
    for topology in TOPOLOGIES:
        for severity in SEVERITIES:
            match = best.get(GridCellKey(topology, severity))
            cells.append(
                {
                    "topology": topology,
                    "severity": severity,
                    "directory": match["directory"] if match else None,
                    "manifest": match["manifest"] if match else None,
                    "experiment_summary": match["experiment_summary"] if match else None,
                }
            )
    return {"topologies": TOPOLOGIES, "severities": SEVERITIES, "cells": cells}
