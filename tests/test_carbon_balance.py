"""Carbon conservation of the carbon Sankey's transformation nodes (2026-11-10).

The study-wide oil node of the PyPSA-Eur-CANEU NW Europe run was 109 Mt short in
2040 and nothing noticed: the imbalance check only looked at the energy graph.
Causes: refined oil at the cost table's intensity while the uses emit at the
model's (PatEx) intensity, refinery CO2 leaving the node without entering it, and
the naphtha feedstock carbon kept in plastics having no edge.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from pypsa2html.extract.emissions import _balance_for_horizon, network_oil_intensity

COST_OIL = 0.2571  # cost table (technology-data)
MODEL_OIL = 0.2639  # what the network's oil uses emit with (PatEx)
COST_GAS = 0.198
MODEL_GAS = 0.2024


def _network():
    """One node; 1 snapshot weighted 1e6 h, so 1 MW = 1 TWh and 1 t/h = 1 Mt."""
    n = pypsa.Network()
    n.set_snapshots([0])
    n.snapshot_weightings.loc[:, :] = 1e6
    n.add("Carrier", ["co2", "oil", "oil primary", "AC", "process emissions",
                      "non-sequestered HVC", "naphtha for industry", "land transport oil"])
    n.add("Bus", "co2 atmosphere", carrier="co2", location="EU")
    n.add("Bus", "EU oil primary", carrier="oil primary", location="EU")
    n.add("Bus", "EU oil", carrier="oil", location="EU")
    n.add("Bus", "A", carrier="AC", location="A")
    n.add("Bus", "A process emissions", carrier="process emissions", location="A")
    n.add("Bus", "A naphtha", carrier="naphtha for industry", location="A")
    n.add("Bus", "A transport", carrier="land transport oil", location="A")
    n.add("Bus", "A HVC", carrier="non-sequestered HVC", location="A")
    n.add("Link", "refinery", bus0="EU oil primary", bus1="EU oil", bus2="co2 atmosphere",
          carrier="oil refining", efficiency=0.95, efficiency2=0.0115)
    n.add("Link", "A transport oil", bus0="EU oil", bus1="A transport", bus2="co2 atmosphere",
          carrier="land transport oil", efficiency2=MODEL_OIL)
    n.add("Link", "A naphtha", bus0="EU oil", bus1="A naphtha", bus2="A process emissions",
          carrier="naphtha for industry", efficiency2=0.0122)
    n.add("Link", "A process", bus0="A process emissions", bus1="co2 atmosphere",
          carrier="process emissions")
    n.add("Link", "A HVC to air", bus0="A HVC", bus1="co2 atmosphere", carrier="HVC to air",
          efficiency=0.0)
    refined = 95.0  # 100 TWh primary -> 95 refined = 70 transport + 25 naphtha
    p = {
        "p0": {"refinery": 100.0, "A transport oil": 70.0, "A naphtha": 25.0,
               "A process": 25 * 0.0122, "A HVC to air": 20.0},
        "p1": {"refinery": -refined, "A transport oil": -70.0, "A naphtha": -25.0,
               "A process": -25 * 0.0122, "A HVC to air": 0.0},
        "p2": {"refinery": -100 * 0.0115, "A transport oil": -70 * MODEL_OIL,
               "A naphtha": -25 * 0.0122, "A process": 0.0, "A HVC to air": 0.0},
    }
    # gas: 10 TWh supplied = 5 burnt in a CCGT + 1 in a CCGT with capture (90 %
    # stored) + 4 used as feedstock (a load, no carbon port)
    n.add("Carrier", ["gas", "co2 stored"])
    n.add("Bus", "A gas", carrier="gas", location="A")
    n.add("Bus", "A co2 stored", carrier="co2 stored", location="A")
    n.add("Generator", "A gas", bus="A gas", carrier="gas")
    n.add("Link", "A CCGT", bus0="A gas", bus1="A", bus2="co2 atmosphere", carrier="CCGT",
          efficiency2=MODEL_GAS)
    n.add("Link", "A CCGT CC", bus0="A gas", bus1="A", bus2="co2 atmosphere",
          bus3="A co2 stored", carrier="CCGT CC", efficiency2=0.1 * MODEL_GAS,
          efficiency3=0.9 * MODEL_GAS)
    n.add("Load", "A gas feedstock", bus="A gas", carrier="gas feedstock for industry",
          p_set=4.0)
    n.generators_t.p = pd.DataFrame({"A gas": [10.0]}, index=n.snapshots)
    p["p0"].update({"A CCGT": 5.0, "A CCGT CC": 1.0})
    p["p1"].update({"A CCGT": -2.5, "A CCGT CC": -0.5})
    p["p2"].update({"A CCGT": -5 * MODEL_GAS, "A CCGT CC": -0.1 * MODEL_GAS})
    p["p3"] = {"A CCGT CC": -0.9 * MODEL_GAS}
    for port, values in p.items():
        setattr(n.links_t, port, pd.DataFrame(values, index=n.snapshots).reindex(
            columns=n.links.index, fill_value=0.0))
    return n


def _ctx(network):
    return SimpleNamespace(
        networks={2050: network},
        _files={},
        config=SimpleNamespace(raw={}),
        component_index=lambda horizon, component, node: getattr(network, component).index,
        cost=lambda tech, param, default=None: {"oil": COST_OIL, "gas": COST_GAS}.get(
            tech, default),
    )


def test_network_oil_intensity_is_the_models_not_the_cost_tables():
    n = _network()
    assert network_oil_intensity(_ctx(n), n, 2050) == pytest.approx(MODEL_OIL)


def test_gas_node_conserves_carbon_capture_and_feedstock_included():
    from pypsa2html.extract.emissions import network_fuel_intensity

    n = _network()
    # capture plants count whole (atmosphere + stored ports)
    assert network_fuel_intensity(_ctx(n), n, 2050, "gas") == pytest.approx(MODEL_GAS)
    b = _balance_for_horizon(_ctx(n), "ALL", 2050).set_index("entry")["value"]
    assert b["fossil gas"] == pytest.approx(10 * MODEL_GAS)
    assert b["gas feedstock carbon"] == pytest.approx(4 * MODEL_GAS)
    capture_plant = sum(v for k, v in b.items() if "CCGT CC" in k)  # residual + stored
    assert capture_plant == pytest.approx(1 * MODEL_GAS)
    assert b["fossil gas"] == pytest.approx(
        b["CCGT"] + capture_plant + b["gas feedstock carbon"])


def test_oil_node_conserves_carbon():
    n = _network()
    b = _balance_for_horizon(_ctx(n), "ALL", 2050).set_index("entry")["value"]
    inflow = b["fossil oil"] + b["fossil oil refinery"]
    outflow = (b["land transport oil emissions"] + b["oil refining emissions"]
               + b["naphtha carbon"])
    assert inflow == pytest.approx(95 * MODEL_OIL + 1.15)
    assert inflow == pytest.approx(outflow)
    # the naphtha node: process share + end of life (0 here) + kept in products
    assert b["naphtha carbon"] == pytest.approx(25 * MODEL_OIL)
    assert b["naphtha carbon in products"] == pytest.approx(25 * (MODEL_OIL - 0.0122))


def test_waste_chp_carbon_comes_from_the_naphtha_node():
    """PyPSA-Eur-CANEU ``patex.waste_to_energy``: 2 TWh of waste burned at 0.357 t/MWh."""
    n = _network()
    n.add("Link", "A waste CHP", bus0="A HVC", bus1="A", bus2="", bus3="co2 atmosphere",
          carrier="waste CHP", efficiency=0.2, efficiency3=0.357)
    for port, value in {"p0": 2.0, "p1": -0.4, "p2": 0.0, "p3": -2 * 0.357}.items():
        getattr(n.links_t, port)["A waste CHP"] = value
    b = _balance_for_horizon(_ctx(n), "ALL", 2050).set_index("entry")["value"]
    assert b["naphtha carbon to waste CHP"] == pytest.approx(2 * 0.357)
    assert b["Waste CHP emissions"] == pytest.approx(2 * 0.357)
    assert b["naphtha carbon in products"] == pytest.approx(
        25 * (MODEL_OIL - 0.0122) - 2 * 0.357)


def test_co2_sequestration_and_pipeline_trade_are_read_on_the_links():
    """Node A captures 3 Mt (process emissions CC), sends 2 Mt to B by pipeline and
    sequesters 1 Mt: sequestration is the sequestration link, not A's residual."""
    from pypsa2html.extract.emissions import _co2_pipeline_trade, _co2_sequestered, _Reductions

    n = _network()
    n.add("Carrier", ["co2 sequestered"])
    n.add("Bus", "B co2 stored", carrier="co2 stored", location="B")
    n.add("Bus", "A co2 sequestered", carrier="co2 sequestered", location="A")
    n.add("Link", "A capture", bus0="A process emissions", bus1="A co2 stored",
          carrier="process emissions CC")
    n.add("Link", "A co2 sequestered", bus0="A co2 stored", bus1="A co2 sequestered",
          carrier="co2 sequestered")
    n.add("Link", "CO2 pipeline A -> B", bus0="A co2 stored", bus1="B co2 stored",
          carrier="CO2 pipeline", p_min_pu=-1)
    for port, value in {"p0": {"A capture": 3.0, "A co2 sequestered": 1.0,
                               "CO2 pipeline A -> B": 2.0},
                        "p1": {"A capture": -3.0, "A co2 sequestered": -1.0,
                               "CO2 pipeline A -> B": -2.0}}.items():
        for link, v in value.items():
            getattr(n.links_t, port)[link] = v
    ctx = _ctx(n)
    ctx.locations_for = lambda node: {"A": ["A"], "B": ["B"]}.get(node)
    assert _co2_sequestered(ctx, _Reductions(ctx, "A", 2050)) == pytest.approx(1.0)
    assert _co2_pipeline_trade(ctx, "A", 2050) == pytest.approx((2.0, 0.0))
    assert _co2_pipeline_trade(ctx, "B", 2050) == pytest.approx((0.0, 2.0))
    assert _co2_pipeline_trade(ctx, "ALL", 2050) == (0.0, 0.0)


def test_waste_chp_capture_residual_is_booked_once():
    """``waste CHP CC``: the residual (port 3) and the stored part (port 4) add up to the
    naphtha carbon burned; the residual is not also counted as plain-plant emissions."""
    n = _network()
    n.add("Link", "A waste CHP CC", bus0="A HVC", bus1="A", bus2="", bus3="co2 atmosphere",
          bus4="co2 stored", carrier="waste CHP CC", efficiency=0.2, efficiency3=0.036,
          efficiency4=0.321)
    for port, value in {"p0": 2.0, "p1": -0.4, "p2": 0.0, "p3": -2 * 0.036,
                        "p4": -2 * 0.321}.items():
        getattr(n.links_t, port)["A waste CHP CC"] = value
    b = _balance_for_horizon(_ctx(n), "ALL", 2050).set_index("entry")["value"]
    assert b.get("Waste CHP emissions", 0.0) == pytest.approx(0.0)
    assert b["Waste CHP CC1"] == pytest.approx(2 * 0.036)
    assert b["Waste CHP CC"] == pytest.approx(2 * 0.321)
    assert b["naphtha carbon to waste CHP"] == pytest.approx(2 * (0.036 + 0.321))


RUN = Path(__file__).resolve().parents[2] / "PyPSA-Eur-CANEU"
# Validated PyPSA-Eur-CANEU runs (the cap-50 run was deleted 2026-11-18): the NW Europe
# reference (audit #1-#7, 2023 weather) and the EU28 run (bio self-sufficient, biogas
# upgrading and renewable waste capture). Each test skips a run that is not on disk.
RUN_NAMES = (
    "patex-levers2-nwe-bz-co2-90-2040-24h-2023-shift",
    "patex-levers2-eu28-bz-co2-80-2040-24h-2023-shift-bio",
)
_on_disk = pytest.mark.parametrize("RUN_NAME", [
    pytest.param(r, marks=pytest.mark.skipif(not (RUN / "results" / r / "networks").exists(),
                                             reason=f"run {r} not on disk"))
    for r in RUN_NAMES])


@_on_disk
def test_carbon_sankey_transformation_nodes_balance_on_the_nwe_run(tmp_path, RUN_NAME):
    """Carbon transformation nodes conserve carbon: study-wide (measured fossil
    inflows) and per country (EU and non-EU)."""
    import yaml

    from pypsa2html import indicators
    from pypsa2html.config import load_config
    from pypsa2html.context import build_context

    cfg = {
        "root": str(RUN),
        "model": {"network_pattern": "networks/solved_{horizon}.nc", "flow_threshold": 0.01},
        "nodes": {"detect": True, "aggregate": {"enabled": True, "code": "ALL",
                                                "label": "All countries"}},
        "scenarios": [{"name": RUN_NAME, "label": RUN_NAME,
                       "results_dir": f"results/{RUN_NAME}",
                       "resources_dir": f"resources/{RUN_NAME}"}],
        "output": {"dir": str(tmp_path / "html")},
    }
    path = tmp_path / "p2h.yaml"
    path.write_text(yaml.safe_dump(cfg))
    config = load_config(path)
    ctx = build_context(config, RUN_NAME)
    for node in ("ALL", "DE00AC", "GB00AC"):
        flows = indicators.for_node(ctx, node).flows_co2
        imb = indicators.graph_imbalances(flows, ctx.taxonomy,
                                          codes=indicators.CARBON_BALANCE_NODES)
        assert imb.empty, (node, imb.to_dict("records"))


@_on_disk
def test_reported_emissions_equal_the_models_atmosphere(tmp_path, RUN_NAME):
    """The emissions-by-sector total and the carbon Sankey's net equal the CO2 the
    model leaves in the atmosphere (2026-11-10: carbon-neutral import uptakes and
    discovered atmosphere ports were missing, a coal supply generator was counted
    as combustion)."""
    import yaml

    from pypsa2html import indicators
    from pypsa2html.config import load_config
    from pypsa2html.context import build_context

    cfg = {
        "root": str(RUN),
        "model": {"network_pattern": "networks/solved_{horizon}.nc", "flow_threshold": 0.01},
        "nodes": {"detect": True, "aggregate": {"enabled": True, "code": "ALL",
                                                "label": "All countries"}},
        "scenarios": [{"name": RUN_NAME, "label": RUN_NAME,
                       "results_dir": f"results/{RUN_NAME}",
                       "resources_dir": f"resources/{RUN_NAME}"}],
        "output": {"dir": str(tmp_path / "html")},
    }
    path = tmp_path / "p2h.yaml"
    path.write_text(yaml.safe_dump(cfg))
    ctx = build_context(load_config(path), RUN_NAME)
    data = indicators.for_node(ctx, "ALL")
    total = data.ghg_sector.sum(axis=1)
    for year in total.index:
        n = ctx.networks[int(year)]
        atm = n.stores.index[n.stores.carrier == "co2"]
        model = float(n.stores_t.e[atm].iloc[-1].sum()) / 1e6
        assert total[year] == pytest.approx(model, abs=0.5), year
        row = data.flows_co2.loc[year]
        net = sum(v for (a, b, t), v in row.items()
                  if a == "atm" and b in ("net_ghg", "netcap_ghg", "netoth_ghg"))
        assert net == pytest.approx(model, abs=0.5), year
