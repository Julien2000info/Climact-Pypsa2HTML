"""Imports per vector and self-sufficiency per supply region (PyPSA-Eur-CANEU caps)."""

import pandas as pd
import pypsa
import pytest

from pypsa2html.extract.supply_limits import SupplyBalance, regions


def network():
    """BE (PatEx "EU" region) and GB, two snapshots of 12 h; flows set by hand."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2040-01-01", periods=2, freq="12h"))
    n.snapshot_weightings.loc[:, :] = 12.0
    for ct in ("BE", "GB"):
        n.add("Bus", f"{ct}0 0", carrier="AC", country=ct, location=f"{ct}0 0")
        n.add("Bus", f"{ct}0 0 gas", carrier="gas", country=ct, location=f"{ct}0 0")
        n.add("Generator", f"{ct}0 0 gas", bus=f"{ct}0 0 gas", carrier="gas")
        n.add("Generator", f"{ct}0 0 onwind", bus=f"{ct}0 0", carrier="onwind")
    n.add("Link", "GB-BE gas pipeline", bus0="GB0 0 gas", bus1="BE0 0 gas",
          carrier="gas pipeline")
    n.add("Line", "BE-GB", bus0="BE0 0", bus1="GB0 0", x=0.1, s_nom=100)
    n.add("GlobalConstraint", "import limit gas EU", type="import_limit",
          carrier_attribute="gas|EU:BE", constant=24 * 300.0, sense="<=", mu=-5.0)
    n.add("GlobalConstraint", "import limit gas GB", type="import_limit",
          carrier_attribute="gas|GB", constant=24 * 200.0, sense="<=", mu=0.0)
    idx = n.snapshots
    n.generators_t.p = pd.DataFrame(
        {"BE0 0 gas": [100.0, 100.0], "GB0 0 gas": [150.0, 150.0],
         "BE0 0 onwind": [50.0, 50.0], "GB0 0 onwind": [100.0, 100.0]}, index=idx)
    # 200 MW of gas from GB to BE
    n.links_t.p0 = pd.DataFrame({"GB-BE gas pipeline": [200.0, 200.0]}, index=idx)
    n.links_t.p1 = pd.DataFrame({"GB-BE gas pipeline": [-200.0, -200.0]}, index=idx)
    # electricity: BE -> GB 30 MW, then GB -> BE 10 MW
    n.lines_t.p0 = pd.DataFrame({"BE-GB": [30.0, -10.0]}, index=idx)
    n.lines_t.p1 = pd.DataFrame({"BE-GB": [-30.0, 10.0]}, index=idx)
    return n


def test_regions_from_constraints():
    names = [r.name for r in regions(network())]
    assert names == ["EU", "GB"]


def test_imports_pipeline_inflow_and_cap_use():
    b = SupplyBalance(network())
    t = b.table()
    nyears = 24 / 8760
    eu, gb = t.loc[("EU", "gas")], t.loc[("GB", "gas")]
    twh = lambda mwh: mwh / 1e6 / nyears  # noqa: E731
    assert eu["imports from outside the model"] == pytest.approx(twh(2400))
    assert eu["inflow from modelled regions"] == pytest.approx(twh(4800))
    assert eu["cap use [%]"] == pytest.approx(100.0)  # 7200 MWh = cap
    assert eu["cap dual [EUR/MWh]"] == pytest.approx(5.0)
    assert gb["exports to modelled regions"] == pytest.approx(twh(4800))
    assert gb["cap use [%]"] == pytest.approx(75.0)  # 3600 / 4800


def test_electricity_hourly_parts_and_self_sufficiency():
    b = SupplyBalance(network())
    t = b.table()
    nyears = 24 / 8760
    elc = t.loc[("GB", "electricity")]
    assert elc["total imports"] == pytest.approx(30 * 12 / 1e6 / nyears)
    assert elc["exports to modelled regions"] == pytest.approx(10 * 12 / 1e6 / nyears)
    assert elc["max import [GW]"] == pytest.approx(0.03)
    ss = b.self_sufficiency()
    # GB: domestic 2400 MWh wind; imports 3600 gas + 360 elc; exports 4800 gas + 120 elc;
    # a net exporter is above 100 %
    assert ss.at["GB", "self-sufficiency [%]"] == pytest.approx(
        100 * 2400 / (2400 + 3960 - 4920)
    )
    # BE: domestic 1200 wind; imports 2400 + 4800 gas + 120 elc; exports 360 elc
    assert ss.at["EU", "self-sufficiency [%]"] == pytest.approx(
        100 * 1200 / (1200 + 7320 - 360)
    )
