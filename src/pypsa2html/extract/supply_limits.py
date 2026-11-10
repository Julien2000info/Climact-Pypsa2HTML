"""Imports per energy vector and self-sufficiency per supply region (PyPSA-Eur-CANEU).

PyPSA-Eur-CANEU's ``import_limits`` writes one ``GlobalConstraint`` of type
``import_limit`` per fuel and region (``carrier_attribute`` ``"fuel|EU:c1,c2,..."``
for the PatEx region, ``"fuel|GB"`` for a modelled country outside it). This module
evaluates on a solved network what each of those constraints counts, split by origin,
plus the electricity exchanged between the regions (not capped):

- imports from outside the model: the fuel's import components located in the region
  (the EU region also takes the components on EU-wide buses);
- inflow from the other modelled regions: gross gas / H2 pipeline inflow and, for the
  fuels on an EU-wide bus (the "pools": oil, coal, methanol, NH3), the net annual flow
  of each outside country into the pool (EU side) or out of it (that country);
- exports to the other modelled regions: the same flows the other way.

The component lists mirror ``scripts/_import_limits_config.py`` of PyPSA-Eur-CANEU
(``IMPORT_FUELS``, ``PIPELINES``, ``POOLS``) and the pool flow of
``solve_network.pool_net_flow``; keep them in step.

Self-sufficiency of a region, three indicators (each = domestic / (domestic + imports −
exports), TWh; net imports are annual, so hourly netting does not change them):

- **electricity**: domestic = net electricity generation in the region (generators on
  electricity buses, hydro dispatch, conversion links into electricity: thermal plants,
  CHP, nuclear, fuel cells; storage and grid links excluded); imports / exports = the
  parts of the region's hourly net exchange.
- **molecules** (fuels): domestic = primary fuels produced in the region (domestic gas,
  oil, coal, lignite, solid biomass, biogas) plus hydrogen from electrolysis (fuels made
  from it downstream are transformations, not counted again); imports / exports = every
  fuel vector of the table (from outside the model and from the other regions).
- **primary** (both together): domestic primary production (wind, solar, hydro,
  biomass, domestic fossil fuels, nuclear heat at the Eurostat convention) over all
  vectors; ambient heat of heat pumps not counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: fuel -> (component, carriers) counted as import (PyPSA-Eur-CANEU ``IMPORT_FUELS``)
IMPORT_FUELS = {
    "gas": ("Generator", ["gas"]),
    "oil": ("Generator", ["oil primary", "oil"]),
    "coal": ("Generator", ["coal"]),
    "H2": ("Generator", ["import H2"]),
    "NH3": ("Generator", ["import NH3"]),
    "synthetic gas": ("Link", ["import gas"]),
    "synthetic oil": ("Link", ["import oil"]),
    "methanol": ("Link", ["import methanol"]),
    "solid biomass": ("Link", ["solid biomass import"]),
    "biogas": ("Link", ["import biogas"]),
    "bioliquids": ("Link", ["import bioliquids"]),
}
PIPELINES = {
    "gas": ["gas pipeline", "gas pipeline new"],
    "H2": ["H2 pipeline", "H2 pipeline retrofitted"],
}
POOLS = {
    "oil": ["oil primary", "oil"],
    "coal": ["coal", "lignite"],
    "methanol": ["methanol"],
    "NH3": ["NH3"],
}
#: component -> static table attribute
STATIC = {"Generator": "generators", "Link": "links"}
#: carriers of the cross-border electricity branches
ELECTRICITY_LINKS = ["DC"]
#: generator carriers that are not domestic primary production
NOT_DOMESTIC = {c for _, cs in IMPORT_FUELS.values() for c in cs} | {"uranium"}

COLUMNS = [
    "imports from outside the model",
    "inflow from modelled regions",
    "total imports",
    "cap",
    "cap use [%]",
    "cap dual [EUR/MWh]",
    "exports to modelled regions",
]


@dataclass(frozen=True)
class Region:
    name: str  # "EU" or a country code
    countries: frozenset[str]

    @property
    def is_eu(self) -> bool:
        return self.name == "EU"


def regions(n) -> list[Region]:
    """Supply regions of the run, from the ``import_limit`` constraints (EU first)."""
    glcs = n.global_constraints
    if glcs.empty or "import_limit" not in set(glcs.type):
        return []
    found = {}
    for attr in glcs.carrier_attribute[glcs.type == "import_limit"]:
        _, _, region = str(attr).partition("|")
        if not region:
            continue
        if region.startswith("EU:"):
            found["EU"] = Region("EU", frozenset(region[3:].split(",")))
        else:
            found[region] = Region(region, frozenset([region]))
    return sorted(found.values(), key=lambda r: (not r.is_eu, r.name))


def _bus_country(n, buses: pd.Series) -> pd.Series:
    """Country of buses through their location (NaN for EU-wide buses)."""
    location = buses.map(n.buses.location).replace("", np.nan).fillna(buses)
    location = location.where(location.isin(n.buses.index))
    return location.map(n.buses.country).replace("", np.nan)


def component_countries(n, component: str) -> pd.Series:
    """Country of each component: ``location`` (``"EU"``: none), else its first port
    on a located bus (PyPSA-Eur-CANEU ``component_countries``)."""
    static = getattr(n, STATIC[component])
    out = pd.Series(np.nan, index=static.index, dtype=object)
    explicit_eu = pd.Series(False, index=static.index)
    if "location" in static:
        loc = static["location"].fillna("")
        explicit_eu = loc == "EU"
        located = loc.where(loc != "", np.nan).dropna()
        out = _bus_country(n, located).reindex(static.index)
    ports = (
        ["bus"]
        if component == "Generator"
        else ["bus1", "bus0"]
        + sorted(c for c in static if c.startswith("bus") and c[3:].isdigit() and int(c[3:]) > 1)
    )
    for port in ports:
        if port not in static:
            continue
        missing = out.isna() & ~explicit_eu
        if not missing.any():
            break
        out[missing] = _bus_country(n, static.loc[missing, port]).values
    return out.astype(object)


def _in_region(where: pd.Series, region: Region) -> pd.Series:
    return where.isin(region.countries) | (where.isna() if region.is_eu else False)


def _energy(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted sum over snapshots per column [MWh]."""
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.mul(weights, axis=0).sum()


class SupplyBalance:
    """Imports, exports and domestic production per region of one solved network."""

    def __init__(self, n):
        self.n = n
        self.w = n.snapshot_weightings.generators
        self.nyears = float(self.w.sum()) / 8760.0
        self.regions = regions(n)
        self.gen_ct = component_countries(n, "Generator")
        self.link_ct = component_countries(n, "Link")
        self.gen_e = _energy(n.generators_t.p, self.w).reindex(n.generators.index).fillna(0.0)
        self.bus_ct = _bus_country(n, pd.Series(n.buses.index, n.buses.index))
        self._pool_net: dict[str, pd.Series] = {}

    def _twh(self, mwh: float) -> float:
        return float(mwh) / 1e6 / self.nyears

    def _link_out(self, links, port: int = 1) -> pd.Series:
        """Energy delivered at ``bus<port>`` [MWh] (``-p<port>``)."""
        p = getattr(self.n.links_t, f"p{port}")
        return -_energy(p.reindex(columns=links, fill_value=0.0), self.w)

    def pool_net(self, pool: str) -> pd.Series:
        """Net annual flow of each outside country into ``pool`` [MWh]."""
        if pool in self._pool_net:
            return self._pool_net[pool]
        n = self.n
        buses = n.buses.index[n.buses.carrier.isin(POOLS[pool])]
        buses = set(buses[self.bus_ct.reindex(buses).isna().values])
        outside = sorted({c for r in self.regions if not r.is_eu for c in r.countries})
        net = {}
        for c in outside:
            total, seen = 0.0, False
            gens = n.generators.index[n.generators.bus.isin(buses) & self.gen_ct.eq(c).values]
            if len(gens):
                total += self.gen_e[gens].sum()
                seen = True
            for col in sorted(c_ for c_ in n.links if c_.startswith("bus") and c_[3:].isdigit()):
                links = n.links.index[n.links[col].isin(buses) & self.link_ct.eq(c).values]
                if links.empty:
                    continue
                total += self._link_out(links, int(col[3:])).sum()  # -p_k: supply into the bus
                seen = True
            if seen:
                net[c] = total
        self._pool_net[pool] = pd.Series(net, dtype=float)
        return self._pool_net[pool]

    def fuel_row(self, fuel: str, region: Region) -> dict:
        n = self.n
        component, carriers = IMPORT_FUELS[fuel]
        static = getattr(n, STATIC[component])
        index = static.index[static.carrier.isin(carriers)]
        where = (self.gen_ct if component == "Generator" else self.link_ct).reindex(index)
        index = index[_in_region(where, region).values]
        if component == "Generator":
            world = self.gen_e.reindex(index).sum()
        else:
            world = self._link_out(index).sum()
        inflow = outflow = 0.0
        if fuel in PIPELINES:
            pipes = n.links[n.links.carrier.isin(PIPELINES[fuel])]
            c0, c1 = pipes.bus0.map(self.bus_ct), pipes.bus1.map(self.bus_ct)
            in0, in1 = c0.isin(region.countries), c1.isin(region.countries)
            into = pipes.index[~in0 & c0.notna() & in1]
            out_of = pipes.index[in0 & ~in1 & c1.notna()]
            inflow += self._link_out(into).sum()
            outflow += _energy(n.links_t.p0.reindex(columns=out_of, fill_value=0.0), self.w).sum()
        pool = {"oil": "oil", "coal": "coal", "methanol": "methanol", "NH3": "NH3"}.get(fuel)
        if pool:
            net = self.pool_net(pool)
            if region.is_eu:
                inflow += net.clip(lower=0).sum()
                outflow += (-net).clip(lower=0).sum()
            else:
                c = region.name
                inflow += max(0.0, -net.get(c, 0.0))
                outflow += max(0.0, net.get(c, 0.0))
        glcs = n.global_constraints
        name = glcs.index[
            (glcs.type == "import_limit")
            & glcs.carrier_attribute.str.startswith(f"{fuel}|")
            & glcs.carrier_attribute.str.partition("|")[2].map(
                lambda r: (r.startswith("EU:") if region.is_eu else r == region.name)
            )
        ]
        cap = self._twh(glcs.at[name[0], "constant"]) if len(name) else np.nan
        mu = abs(float(glcs.at[name[0], "mu"])) if len(name) else np.nan
        total = self._twh(world + inflow)
        return {
            "imports from outside the model": self._twh(world),
            "inflow from modelled regions": self._twh(inflow),
            "total imports": total,
            "cap": cap,
            "cap use [%]": 100.0 * total / cap if cap and cap > 1e-9 else np.nan,
            "cap dual [EUR/MWh]": mu,
            "exports to modelled regions": self._twh(outflow),
        }

    def electricity_row(self, region: Region) -> dict:
        """Hourly net exchange of the region over AC lines and DC links: imports and
        exports are its positive and negative parts [TWh/a]; max import in GW."""
        n = self.n
        net = pd.Series(0.0, index=n.snapshots)
        branches = [("lines", n.lines)]
        branches.append(("links", n.links[n.links.carrier.isin(ELECTRICITY_LINKS)]))
        for kind, df in branches:
            if df.empty:
                continue
            c0, c1 = df.bus0.map(self.bus_ct), df.bus1.map(self.bus_ct)
            in0, in1 = c0.isin(region.countries), c1.isin(region.countries)
            t = getattr(n, f"{kind}_t")
            at1 = df.index[~in0 & c0.notna() & in1]  # enters the region at bus1
            at0 = df.index[in0 & ~in1 & c1.notna()]  # enters the region at bus0
            net -= t.p1.reindex(columns=at1, fill_value=0.0).sum(axis=1)
            net -= t.p0.reindex(columns=at0, fill_value=0.0).sum(axis=1)
        imports = float(net.clip(lower=0).mul(self.w).sum())
        exports = float((-net).clip(lower=0).mul(self.w).sum())
        return {
            "imports from outside the model": 0.0,
            "inflow from modelled regions": self._twh(imports),
            "total imports": self._twh(imports),
            "cap": np.nan,
            "cap use [%]": np.nan,
            "cap dual [EUR/MWh]": np.nan,
            "exports to modelled regions": self._twh(exports),
            "max import [GW]": float(net.max()) / 1e3 if len(net) else 0.0,
        }

    def domestic(self, region: Region) -> float:
        """Domestic primary production [TWh/a]."""
        n = self.n
        gens = n.generators.index[
            ~n.generators.carrier.isin(NOT_DOMESTIC)
            & ~n.generators.carrier.str.contains("vent|load", regex=True)
        ]
        gens = gens[_in_region(self.gen_ct.reindex(gens), region).values]
        total = self.gen_e.reindex(gens).clip(lower=0).sum()
        su = n.storage_units
        if not su.empty and not n.storage_units_t.inflow.empty:
            ct = su.bus.map(self.bus_ct)
            units = su.index[_in_region(ct, region).values]
            total += _energy(n.storage_units_t.inflow.reindex(columns=units, fill_value=0.0),
                             self.w).sum()
        uranium = n.buses.index[n.buses.carrier == "uranium"]
        reactors = n.links.index[n.links.bus0.isin(uranium)]
        reactors = reactors[_in_region(self.link_ct.reindex(reactors), region).values]
        total += _energy(n.links_t.p0.reindex(columns=reactors, fill_value=0.0), self.w).sum()
        return self._twh(total)

    def electricity_generation(self, region: Region) -> float:
        """Net electricity generation in the region [TWh/a]."""
        n = self.n
        elec = {"AC", "low voltage"}
        bus_carrier = n.buses["carrier"]
        gens = n.generators.index[
            n.generators.bus.map(bus_carrier).isin(elec)
            & ~n.generators.carrier.str.contains("vent|load|shedding", regex=True)
        ]
        gens = gens[_in_region(self.gen_ct.reindex(gens), region).values]
        total = self.gen_e.reindex(gens).clip(lower=0).sum()
        su = n.storage_units
        if not su.empty:
            hydro = su.index[(su.carrier == "hydro")
                             & _in_region(su.bus.map(self.bus_ct), region).values]
            if len(hydro):
                total += _energy(n.storage_units_t.p.reindex(columns=hydro, fill_value=0.0)
                                 .clip(lower=0), self.w).sum()
        links = n.links
        conv = links.index[
            links.bus1.map(bus_carrier).isin(elec)
            & ~links.bus0.map(bus_carrier).isin(elec)
            & ~links.carrier.str.contains("battery|V2G|discharger|EV", regex=True)
        ]
        conv = conv[_in_region(self.link_ct.reindex(conv), region).values]
        total += self._link_out(conv).clip(lower=0).sum()
        return self._twh(total)

    def molecules_domestic(self, region: Region) -> float:
        """Primary fuels produced in the region plus electrolytic hydrogen [TWh/a]."""
        n = self.n
        elec = {"AC", "low voltage"}
        gens = n.generators.index[
            ~n.generators.bus.map(n.buses["carrier"]).isin(elec)
            & ~n.generators.carrier.isin(NOT_DOMESTIC)
            & ~n.generators.carrier.str.contains("vent|load|shedding|solar thermal|heat",
                                                regex=True)
        ]
        gens = gens[_in_region(self.gen_ct.reindex(gens), region).values]
        total = self.gen_e.reindex(gens).clip(lower=0).sum()
        el = n.links.index[n.links.carrier == "H2 Electrolysis"]
        el = el[_in_region(self.link_ct.reindex(el), region).values]
        total += self._link_out(el).sum()
        return self._twh(total)

    def electricity_flows(self) -> pd.DataFrame:
        """Gross annual electricity flows between regions [TWh/a], sender x receiver,
        measured at the sending end of each line / DC link (hour by hour, per branch)."""
        n = self.n
        region_of = {c: r.name for r in self.regions for c in r.countries}
        names = [r.name for r in self.regions]
        out = pd.DataFrame(0.0, index=names, columns=names)
        for kind, df in (("lines", n.lines),
                         ("links", n.links[n.links.carrier.isin(ELECTRICITY_LINKS)])):
            if df.empty:
                continue
            t = getattr(n, f"{kind}_t")
            r0 = df.bus0.map(self.bus_ct).map(region_of)
            r1 = df.bus1.map(self.bus_ct).map(region_of)
            cross = df.index[r0.notna() & r1.notna() & (r0 != r1)]
            for i in cross:
                p0 = t.p0[i] if i in t.p0 else pd.Series(0.0, index=n.snapshots)
                out.at[r0[i], r1[i]] += float(p0.clip(lower=0).mul(self.w).sum())
                out.at[r1[i], r0[i]] += float((-p0).clip(lower=0).mul(self.w).sum())
        return out / 1e6 / self.nyears

    def table(self) -> pd.DataFrame:
        """Rows (region, vector); columns :data:`COLUMNS` [TWh/a]."""
        rows = {}
        for region in self.regions:
            for fuel in IMPORT_FUELS:
                rows[(region.name, fuel)] = self.fuel_row(fuel, region)
            rows[(region.name, "electricity")] = self.electricity_row(region)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).T
        df.index.names = ["region", "vector"]
        return df

    def self_sufficiency(self) -> pd.DataFrame:
        """Per region: domestic, imports, exports [TWh/a] and self-sufficiency [%] for
        electricity, molecules and primary energy (see the module docstring)."""
        table = self.table()

        def ratio(domestic, imports, exports):
            supply = domestic + imports - exports
            return 100.0 * domestic / supply if supply > 1e-6 else np.nan

        out = {}
        for region in self.regions:
            t = table.loc[region.name]
            elc = t.loc["electricity"]
            fuels = t.drop(index="electricity")
            row = {}
            for kind, domestic, imports, exports in (
                ("electricity", self.electricity_generation(region),
                 float(elc["total imports"]), float(elc["exports to modelled regions"])),
                ("molecules", self.molecules_domestic(region),
                 float(fuels["total imports"].sum()),
                 float(fuels["exports to modelled regions"].sum())),
                ("primary", self.domestic(region), float(t["total imports"].sum()),
                 float(t["exports to modelled regions"].sum())),
            ):
                row[f"{kind} domestic"] = domestic
                row[f"{kind} imports"] = imports
                row[f"{kind} exports"] = exports
                row[f"{kind} self-sufficiency [%]"] = ratio(domestic, imports, exports)
            out[region.name] = row
        return pd.DataFrame(out).T
