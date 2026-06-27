"""
Adaptive Public-Transport Control — interactive dashboard.

A single screen that demonstrates the whole method end to end:

  1. pick an operating context (time of day, day of week, optional cause);
  2. the Bayesian Belief Network returns P(overload | context);
  3. the decision engine maps that probability onto a control action (a0..a5);
  4. the risk-aware A* plans a route on the Prague network and the map shows how
     a higher risk sensitivity (lambda) trades a little length for less overload.

Run with:  transit-control serve     (or  streamlit run transit/app/dashboard.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when Streamlit runs this file as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from transit import config
from transit.bbn import BBNModel
from transit.routing import TransportGraph
from transit.decision import DecisionEngine, ACTIONS


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading Bayesian network ...")
def load_model() -> BBNModel:
    return BBNModel.load(config.OUT_BBN_MODEL)


@st.cache_resource(show_spinner="Building Prague transport graph ...")
def load_graph() -> TransportGraph:
    return TransportGraph.from_enriched(config.OUT_STOPS_ENRICHED, verbose=False)


def _risk_color(p: float):
    """Green (safe) -> red (overloaded) RGB for a probability in [0, 1]."""
    p = max(0.0, min(1.0, float(p)))
    return [int(40 + 215 * p), int(180 * (1 - p)), 70]


# --------------------------------------------------------------------------- #
# BBN structure as Graphviz DOT (no extra dependency — Streamlit ships it)
# --------------------------------------------------------------------------- #

def bbn_dot(highlight: dict, p_overload: float) -> str:
    palette = {
        "TimeOfDay": "#3B7EAA", "DayOfWeek": "#3B7EAA", "Boro": "#3B7EAA",
        "School_Age_or_PreK": "#3B7EAA", "Reason": "#F4A261",
        "overload": "#D62828", "How_Long_Delayed": "#2A9D8F",
        "Has_Contractor_Notified_Schools": "#2A9D8F",
    }
    lines = [
        "digraph BBN {",
        "  rankdir=TB; bgcolor=transparent;",
        '  node [style="filled,rounded", shape=box, fontname="Helvetica", '
        'fontcolor=white, color="#1E3A5F", penwidth=1.5];',
        '  edge [color="#1E3A5F", penwidth=1.3];',
    ]
    for node in config.BBN_NODES:
        label = node
        if node == "overload":
            label = f"overload\\nP={p_overload:.2f}"
        elif node in highlight:
            label = f"{node}\\n= {highlight[node]}"
        border = ', penwidth=3, color="#111111"' if node in highlight or node == "overload" else ""
        lines.append(f'  "{node}" [label="{label}", fillcolor="{palette[node]}"{border}];')
    for a, b in config.BBN_EDGES:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Map rendering (pydeck if available, otherwise a plain st.map fallback)
# --------------------------------------------------------------------------- #

def _render_map(graph: TransportGraph, base: dict, risk: dict) -> None:
    records = graph.stops_records()
    try:
        import pydeck as pdk
    except ImportError:
        import pandas as pd
        st.info("Install `pydeck` for the coloured map + drawn route "
                "(pip install -r requirements-app.txt). Showing stops only.")
        st.map(pd.DataFrame([{"lat": r["lat"], "lon": r["lon"]} for r in records]))
        return

    # Scatter of every stop, coloured by overload probability.
    scatter_data = [
        {"lon": r["lon"], "lat": r["lat"], "name": r["name"],
         "p": round(r["p_overload"], 3), "color": _risk_color(r["p_overload"])}
        for r in records
    ]
    scatter = pdk.Layer(
        "ScatterplotLayer", data=scatter_data,
        get_position="[lon, lat]", get_fill_color="color",
        get_radius=35, opacity=0.55, pickable=True,
    )

    # Distance-only baseline (grey) vs risk-aware route (blue).
    path_layers = []
    if base["reachable"]:
        path_layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": graph.path_coords(base["path"]), "name": "λ=0 (shortest)"}],
            get_path="path", get_color=[150, 150, 150], width_min_pixels=4,
            get_width=5, pickable=True,
        ))
    if risk["reachable"]:
        path_layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": graph.path_coords(risk["path"]),
                   "name": f"λ={risk['lambda']} (risk-aware)"}],
            get_path="path", get_color=[30, 110, 220], width_min_pixels=5,
            get_width=7, pickable=True,
        ))

    # Mark origin and destination.
    endpoints = []
    for sid, tag in ((risk["origin"], "origin"), (risk["dest"], "dest")):
        s = graph.stop(sid)
        endpoints.append({"lon": s["lon"], "lat": s["lat"], "name": tag})
    endpoint_layer = pdk.Layer(
        "ScatterplotLayer", data=endpoints,
        get_position="[lon, lat]", get_fill_color=[20, 20, 20],
        get_radius=90, pickable=True,
    )

    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    view = pdk.ViewState(
        latitude=sum(lats) / len(lats), longitude=sum(lons) / len(lons),
        zoom=10.5, pitch=0,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[scatter, *path_layers, endpoint_layer],
        initial_view_state=view,
        map_style=None,
        tooltip={"text": "{name}\nP(overload)={p}"},
    ))
    st.caption("Dots = stops (green safe → red overloaded). "
               "Grey line = shortest route (λ=0); blue line = risk-aware route.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Adaptive Transport Control", page_icon="🚌",
                   layout="wide")

st.title("🚌 Adaptive Public-Transport Control")
st.caption("Bayesian Belief Network · risk-aware A\\* · adaptive decision module "
           "— trained on NY Bus Breakdown, applied to Prague.")

# Fail gracefully if the pipeline hasn't been run yet.
if not config.OUT_BBN_MODEL.exists() or not config.OUT_STOPS_ENRICHED.exists():
    st.error(
        "Model artefacts not found. Build them first:\n\n"
        "```\ntransit-control run --fetch\n```\n\n"
        f"Expected:\n- {config.OUT_BBN_MODEL}\n- {config.OUT_STOPS_ENRICHED}"
    )
    st.stop()

model = load_model()
graph = load_graph()

# ---------- Sidebar: context + routing controls ---------- #
with st.sidebar:
    st.header("Operating context")
    opts = model.context_options(["TimeOfDay", "DayOfWeek", "Reason"])

    tod = st.selectbox("Time of day", opts.get("TimeOfDay", ["peak_morning"]),
                       index=0)
    dow_states = opts.get("DayOfWeek", ["weekday"])
    dow = st.selectbox("Day of week", dow_states, index=0)

    use_cause = st.checkbox("Fix a failure cause", value=False,
                            help="Off = the BBN marginalises over all causes.")
    cause = None
    if use_cause and "Reason" in opts:
        cause = st.selectbox("Reason", opts["Reason"], index=0)

    st.divider()
    st.header("Routing")
    lam = st.slider("Risk sensitivity λ", 0.0, 10.0, 2.0, 0.5,
                    help="f*(n) = g(n) + h(n) + λ·R(n)")
    free_units = st.checkbox("Spare vehicles available", value=True)

context = {"TimeOfDay": tod, "DayOfWeek": dow}
if cause:
    context["Reason"] = cause

p_overload = model.predict_overload(context)
decision = DecisionEngine().decide(p_overload, free_units_available=free_units)

# ---------- Top row: probability + decision + BBN ---------- #
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Overload probability")
    band_color = "#2A9D8F" if p_overload < 0.5 else ("#F4A261" if p_overload < 0.7 else "#D62828")
    st.markdown(
        f"<div style='font-size:64px;font-weight:700;color:{band_color};"
        f"line-height:1'>{p_overload:.0%}</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, p_overload))
    st.caption(f"P(overload | {context})")

    if cause is None:
        causes = model.reason_given(context)
        if causes:
            top = sorted(causes.items(), key=lambda kv: kv[1], reverse=True)[:3]
            st.write("Most likely causes for this context:")
            for name, prob in top:
                st.write(f"• {name} — {prob:.0%}")

    st.subheader("Recommended action")
    primary = decision.primary
    st.markdown(f"**{primary.code} — {primary.name}**")
    st.write(primary.description)
    if decision.supporting:
        st.write("In parallel: " +
                 ", ".join(f"{a.code} ({a.name})" for a in decision.supporting))
    if decision.reserve:
        extra = (f" (escalate after {decision.escalate_after_min} min)"
                 if decision.escalate_after_min else "")
        st.caption(f"Held in reserve: {decision.reserve.code} — {decision.reserve.name}{extra}")
    st.caption(f"Band {decision.band} · relative cost {decision.total_cost}")

with right:
    st.subheader("Bayesian Belief Network")
    st.graphviz_chart(bbn_dot(context, p_overload))
    st.caption("Blue = inputs · orange = Reason · red = overload · "
               "teal = consequences. Highlighted nodes are fixed by the context.")

st.divider()

# ---------- Map + route ---------- #
st.subheader("Risk-aware routing on the Prague network")

named = graph.named_stops(limit=400)
if len(named) < 2:
    st.info("Not enough named, connected stops to plan a route.")
    st.stop()

labels = {f"{s.get('name') or s['id']}  ·  {s['id']}": s["id"] for s in named}
keys = list(labels.keys())

c1, c2 = st.columns(2)
with c1:
    o_label = st.selectbox("Origin", keys, index=0)
with c2:
    d_label = st.selectbox("Destination", keys, index=min(7, len(keys) - 1))

origin, dest = labels[o_label], labels[d_label]

base = graph.route(origin, dest, lambda_risk=0.0)        # distance-only baseline
risk = graph.route(origin, dest, lambda_risk=lam)        # risk-aware

if not base["reachable"] or not risk["reachable"]:
    st.warning("These two stops are in different graph components — no route exists. "
               "Pick a closer pair.")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Length (λ=0 → λ)", f"{risk['length_km']:.2f} km",
              f"{risk['length_km'] - base['length_km']:+.2f} km")
    m2.metric("Mean P(overload)", f"{risk['r_overload']:.3f}",
              f"{risk['r_overload'] - base['r_overload']:+.3f}", delta_color="inverse")
    m3.metric("Stops on route", f"{risk['n_stops']}",
              f"{risk['n_stops'] - base['n_stops']:+d}")

    _render_map(graph, base, risk)
