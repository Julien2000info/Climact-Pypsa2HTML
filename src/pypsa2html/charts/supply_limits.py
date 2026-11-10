"""Imports per energy vector and self-sufficiency per supply region.

Builders for PyPSA-Eur-CANEU runs with ``import_limits`` (see
:mod:`pypsa2html.extract.supply_limits`). The supply regions (the PatEx "EU" region and
each modelled country outside it) cut across the report's nodes: both sections show
the same region-level content on every node (so that the page exists wherever the
navigation links to it), and only when the networks carry the ``import_limit``
constraints.
"""

from __future__ import annotations

import html
import logging

import pandas as pd
import plotly.graph_objects as go

from ..extract.supply_limits import COLUMNS, IMPORT_FUELS, SupplyBalance
from .base import Html, _layout
from .keyfigures import fmt

logger = logging.getLogger(__name__)

#: line colours per region (EU first, then the countries in order)
REGION_COLOURS = ["#1f4e9c", "#c0392b", "#16a085", "#8e44ad", "#d68910", "#566573"]


def _balances(ctx) -> dict[int, SupplyBalance] | None:
    """One :class:`SupplyBalance` per horizon (cached), ``None`` without import caps."""
    key = ("supply_limits",)
    if key not in ctx._files:
        out = {}
        for year in ctx.horizons:
            try:
                b = SupplyBalance(ctx.networks[int(year)])
            except Exception as err:  # noqa: BLE001 - a missing horizon is not fatal
                logger.warning("supply limits @ %s: %s", year, err)
                continue
            if b.regions:
                out[int(year)] = b
        ctx._files[key] = out or None
    return ctx._files[key]


def self_sufficiency_regions(ctx, node: str, section) -> go.Figure | None:
    """Self-sufficiency [%] of each supply region over the horizons."""
    balances = _balances(ctx)
    if not balances:
        return None
    frame = pd.DataFrame(
        {y: b.self_sufficiency()["self-sufficiency [%]"] for y, b in balances.items()}
    ).T
    fig = go.Figure()
    for i, region in enumerate(frame.columns):
        fig.add_scatter(
            x=list(frame.index), y=frame[region].round(1), mode="lines+markers",
            name=region, line={"color": REGION_COLOURS[i % len(REGION_COLOURS)]},
        )
    fig.add_hline(y=100, line_dash="dash", line_color="#888888",
                  annotation_text="self-sufficient", annotation_position="top left")
    _layout(fig, title="", unit=section.unit or "%", years=list(frame.index))
    return fig


def _summary_table(balances: dict[int, SupplyBalance]) -> str:
    years = list(balances)
    tables = {y: b.self_sufficiency() for y, b in balances.items()}
    rows = []
    measures = [
        ("domestic production", "Domestic primary production", "TWh/a"),
        ("imports", "Imports (all vectors, incl. from modelled regions)", "TWh/a"),
        ("exports", "Exports to modelled regions", "TWh/a"),
        ("self-sufficiency [%]", "Self-sufficiency", "%"),
        ("electricity net import", "Electricity net import", "TWh/a"),
    ]
    for region in tables[years[0]].index:
        rows.append(f'<tr class="p2h-group"><td colspan="{len(years) + 2}">'
                    f"{html.escape(region)}</td></tr>")
        for key, label, unit in measures:
            cells = "".join(
                f'<td class="num">{fmt(tables[y].at[region, key], 1)}</td>' for y in years
            )
            rows.append(f'<tr><td class="p2h-indent">{label}</td>'
                        f'<td class="p2h-unit-col">{unit}</td>{cells}</tr>')
    head = "".join(f'<th class="num">{y}</th>' for y in years)
    return (f'<table class="p2h-table"><thead><tr><th>Region</th><th>Unit</th>{head}'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _vector_table(balance: SupplyBalance) -> str:
    table = balance.table()
    cols = COLUMNS + ["max import [GW]"]
    head = "".join(f'<th class="num">{html.escape(c)}</th>' for c in cols)
    rows = []
    for region in dict.fromkeys(table.index.get_level_values("region")):
        rows.append(f'<tr class="p2h-group"><td colspan="{len(cols) + 1}">'
                    f"{html.escape(region)}</td></tr>")
        for vector, values in table.loc[region].iterrows():
            if vector != "electricity" and (
                values[["total imports", "exports to modelled regions"]].abs().max() < 1e-3
                and not values["cap"] > 1e-3
            ):
                continue  # nothing imported, exported or allowed
            decimals = {"cap use [%]": 0, "cap dual [EUR/MWh]": 2}
            cells = "".join(
                f'<td class="num">{fmt(values.get(c), decimals.get(c, 1))}</td>' for c in cols
            )
            rows.append(f'<tr><td class="p2h-indent">{html.escape(vector)}</td>{cells}</tr>')
    return (f'<table class="p2h-table"><thead><tr><th>Vector</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>")


def imports_by_vector(ctx, node: str, section) -> Html | None:
    """Summary per region, then imports, caps and exports per vector and horizon."""
    balances = _balances(ctx)
    if not balances:
        return None
    parts = [_summary_table(balances)]
    for i, (year, balance) in enumerate(balances.items()):
        parts.append(
            f"<details{' open' if i == 0 else ''}><summary>{year}: imports per energy "
            f"vector [TWh/a]</summary>{_vector_table(balance)}</details>"
        )
    vectors = ", ".join(IMPORT_FUELS)
    parts.append(
        '<p class="p2h-note">Regions of the import caps: <b>EU</b> = the PatEx countries '
        "of the run together; each other modelled country on its own. Imports from "
        "outside the model enter at the region's own import components (world price). "
        "Inflow from / exports to modelled regions: gross gas and H2 pipeline flows "
        "across the region's border and, for oil, coal, methanol and NH3 (one EU-wide "
        "bus), the net annual flow of each non-EU country into or out of that bus. "
        "Capped vectors: " + html.escape(vectors) + ". Electricity is not capped: "
        "imports and exports are the positive and negative parts of the region's hourly "
        "net exchange over AC lines and DC links. Self-sufficiency = domestic primary "
        "production / (domestic production + imports − exports); domestic production = "
        "wind, solar, hydro, biomass, domestic gas, oil, coal and lignite, nuclear heat "
        "(Eurostat convention); ambient heat of heat pumps is not counted. The cap dual "
        "is the marginal value of one more MWh of import allowance.</p>"
    )
    return Html("".join(parts))
