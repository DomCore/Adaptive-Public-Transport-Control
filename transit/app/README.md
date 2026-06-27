# Interactive dashboard

A single-screen Streamlit application that demonstrates the whole adaptive
control method end to end: choose an operating context, watch the Bayesian
Belief Network estimate the overload probability, see the decision engine pick a
control action, and plan a risk-aware route on the Prague stop network.

It is the visual face of the `transit` package — the same library that powers
the `transit-control` CLI and the numbered pipeline. The dashboard never
re-implements anything: it loads the fitted model (`output/bbn_model.pkl`) and
the enriched stops (`output/stops_enriched.json`) and calls `BBNModel`,
`TransportGraph` and `DecisionEngine` directly.

![Dashboard overview](../../docs/img/dashboard-overview.jpg)

*The sidebar selects the operating context and the risk sensitivity λ; the map
compares the shortest route (grey) against the risk-aware route (blue) on the
Prague network, with stops coloured green→red by overload probability.*

## Run it

```bash
# one-time: install the dashboard extras (Streamlit + pydeck)
pip install -e ".[app]"          # or: pip install -r requirements-app.txt

# launch (opens http://localhost:8501)
transit-control serve

# options
transit-control serve --port 8600     # choose a port
transit-control serve --headless      # don't auto-open a browser
```

Equivalent direct invocation, if you prefer not to use the CLI:

```bash
streamlit run transit/app/dashboard.py
```

> **Prerequisite.** The dashboard needs the pipeline artefacts to exist. If they
> are missing it shows a build hint instead of crashing. Build them once with:
> ```bash
> transit-control run --fetch
> ```

## What's on the screen

```
┌───────────────┬───────────────────────────────────────────────┐
│  SIDEBAR      │  Overload probability        Bayesian network  │
│               │   ┌─────────┐                ┌──────────────┐  │
│ Context       │   │  47 %   │  recommended   │  DAG with the │  │
│  · time       │   └─────────┘  action a3+a5  │  context      │  │
│  · day        │                              │  highlighted  │  │
│  · cause      │                              └──────────────┘  │
│               ├───────────────────────────────────────────────┤
│ Routing       │  Risk-aware routing on the Prague network      │
│  · λ slider   │   Origin ▾   Destination ▾                     │
│  · free units │   length Δ | mean P(overload) Δ | stops Δ      │
│               │   ┌───────────────────────────────────────┐    │
│               │   │  map: stops green→red, two routes      │    │
│               │   └───────────────────────────────────────┘    │
└───────────────┴───────────────────────────────────────────────┘
```

### Sidebar — controls

| Control | Meaning |
|---------|---------|
| **Time of day** | `TimeOfDay` evidence for the BBN (`night`, `peak_morning`, `midday`, `peak_evening`, `evening`). |
| **Day of week** | `DayOfWeek` evidence. |
| **Fix a failure cause** | When off, the BBN marginalises over all causes; when on, you pin a specific `Reason`. |
| **Risk sensitivity λ** | The weight in `f*(n) = g(n) + h(n) + λ·R(n)`. λ = 0 is shortest-path; higher λ avoids high-overload stops. |
| **Spare vehicles available** | Feeds the decision engine: with no free units the mid band falls back from *increase frequency* (a1) to *reallocate* (a2). |

### Main panel

1. **Overload probability** — `P(overload | context)` from the BBN, colour-coded
   (teal < 0.5, amber < 0.7, red ≥ 0.7), with the most likely failure causes for
   the chosen time/day when no cause is fixed.
2. **Recommended action** — the control action (a0–a5) the decision engine
   selects for that probability, with any supporting and reserve actions, the
   threshold band and the relative cost.
3. **Bayesian network** — the eight-node DAG (Graphviz). Blue = inputs, orange =
   `Reason`, red = `overload`, teal = consequence nodes; the nodes fixed by your
   context are outlined.
4. **Risk-aware routing** — pick an origin and a destination; three metrics show
   how the risk-aware route (current λ) differs from the shortest route (λ = 0)
   in length, mean overload probability and stop count. The map draws every stop
   coloured green→red by overload probability, the shortest route as a grey line
   and the risk-aware route as a blue line, with origin/destination marked.

## How it maps to the method

| On screen | Code | Dissertation |
|-----------|------|--------------|
| Overload probability | `transit.bbn.BBNModel.predict_overload` | §2.2 |
| Bayesian network DAG | `transit.config.BBN_NODES` / `BBN_EDGES` | §2.2.2 |
| Recommended action | `transit.decision.DecisionEngine` | §2.4 |
| Map + two routes | `transit.routing.TransportGraph` (risk-aware A\*) | §2.1.3 |
| Cross-domain stops | `output/stops_enriched.json` (NY→Prague transfer) | §3.4.5 |

## Notes & troubleshooting

* **First interaction is slow (~a few seconds).** The full 4349-stop Prague
  graph is built on first use, then cached for the session
  (`@st.cache_resource`), so subsequent changes are instant.
* **The map shows only dots, no coloured route.** `pydeck` is not installed —
  run `pip install -e ".[app]"`. Without it the app falls back to a plain
  `st.map`.
* **"Model artefacts not found".** Run the pipeline first:
  `transit-control run --fetch`.
* **Two stops give "no route exists".** They are in different graph components
  (the 0.5 km edge threshold can leave outlying stops disconnected) — pick a
  closer pair.
* **Reproducibility.** Clustering, OD sampling and the train/test split all use
  `random_state = 42`, so the same context and stop pair always yield the same
  numbers.
