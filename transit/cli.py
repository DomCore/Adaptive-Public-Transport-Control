"""
transit.cli — the ``transit-control`` command-line tool.

Turns the pipeline from a pile of numbered scripts into one coherent instrument:

    transit-control info                      # what's built, model summary
    transit-control run --fetch               # run the whole pipeline (steps 0-5)
    transit-control predict --time peak_morning --cause "Heavy Traffic"
    transit-control route --origin <id> --dest <id> --lambda 2.0
    transit-control decide --p 0.78
    transit-control serve                     # launch the Streamlit dashboard

Heavy steps load their artefacts lazily, so ``transit-control --help`` and
``decide`` stay instant even before the pipeline has been run.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from transit import __version__
from transit import config

app = typer.Typer(
    add_completion=False,
    help="Adaptive public-transport control — BBN + risk-aware A* + decisions.",
    no_args_is_help=True,
)


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _require(path: Path, hint: str) -> None:
    """Exit with a friendly message if a required artefact is missing."""
    if not path.exists():
        typer.secho(f"Missing artefact: {path.name}", fg=typer.colors.RED, err=True)
        typer.secho(f"  -> {hint}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #

@app.command()
def info() -> None:
    """Show version, artefact status and a one-line BBN summary."""
    _echo(f"transit-control {__version__}")
    _echo(f"repo root: {config.BASE_DIR}")
    _echo("")
    artefacts = [
        ("BBN model", config.OUT_BBN_MODEL),
        ("BBN meta", config.OUT_BBN_META),
        ("Enriched stops", config.OUT_STOPS_ENRICHED),
        ("Experiment results", config.OUT_RESULTS_AGG),
        ("Input NY CSV", config.INPUT_CSV),
        ("Prague stops", config.INPUT_STOPS_GEOJSON),
    ]
    _echo("Artefacts:")
    for label, path in artefacts:
        mark = "OK " if path.exists() else "-- "
        _echo(f"  [{mark}] {label:20s} {path.name}")

    if config.OUT_BBN_META.exists():
        import json
        meta = json.loads(config.OUT_BBN_META.read_text(encoding="utf-8"))
        _echo("")
        _echo(f"BBN: {meta.get('n_nodes')} nodes, {meta.get('n_edges')} edges, "
              f"valid={meta.get('valid')}, smoke_test={meta.get('smoke_test_passed')}")


# --------------------------------------------------------------------------- #
# run (orchestrate the numbered pipeline)
# --------------------------------------------------------------------------- #

@app.command()
def run(
    fetch: bool = typer.Option(False, "--fetch", help="download Prague stops first (step 0)."),
    from_step: int = typer.Option(1, "--from", help="first step index to run."),
    to_step: int = typer.Option(5, "--to", help="last step index to run."),
) -> None:
    """Run the core pipeline (steps 00-05) via run_all.py."""
    runner = config.BASE_DIR / "run_all.py"
    _require(runner, "run_all.py should sit at the repo root.")
    cmd = [sys.executable, str(runner), "--from", str(from_step), "--to", str(to_step)]
    if fetch:
        cmd.append("--fetch")
    _echo(f"$ {' '.join(cmd)}")
    raise typer.Exit(code=subprocess.run(cmd, cwd=config.BASE_DIR).returncode)


# --------------------------------------------------------------------------- #
# predict (BBN inference)
# --------------------------------------------------------------------------- #

@app.command()
def predict(
    time: str = typer.Option("peak_morning", "--time", "-t",
                             help="TimeOfDay state (e.g. peak_morning, midday)."),
    day: str = typer.Option("weekday", "--day", "-d", help="DayOfWeek state."),
    cause: Optional[str] = typer.Option(None, "--cause", "-c",
                                        help="Reason state; omit to marginalise over causes."),
    decide_too: bool = typer.Option(True, "--decide/--no-decide",
                                    help="also show the recommended control action."),
) -> None:
    """Query P(overload | context) from the fitted BBN."""
    _require(config.OUT_BBN_MODEL, "Build it first: transit-control run --to 2")
    from transit.bbn import BBNModel

    model = BBNModel.load(config.OUT_BBN_MODEL)
    context = {"TimeOfDay": time, "DayOfWeek": day}
    if cause:
        context["Reason"] = cause

    # Validate the evidence against the model's known states.
    for node, value in context.items():
        valid = model.states(node)
        if value not in valid:
            typer.secho(f"Unknown {node} state: {value!r}", fg=typer.colors.RED, err=True)
            typer.secho(f"  valid: {', '.join(valid)}", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(code=1)

    p = model.predict_overload(context)
    _echo(f"context: {context}")
    _echo(f"P(overload) = {p:.4f}")

    if cause is None:
        # Surface the most likely cause for this time/day as colour.
        causes = model.reason_given(context)
        if causes:
            top = sorted(causes.items(), key=lambda kv: kv[1], reverse=True)[:3]
            _echo("most likely causes: " +
                  ", ".join(f"{name} ({prob:.2f})" for name, prob in top))

    if decide_too:
        from transit.decision import DecisionEngine
        d = DecisionEngine().decide(p)
        _echo(f"decision  = {d}")


# --------------------------------------------------------------------------- #
# route (risk-aware A*)
# --------------------------------------------------------------------------- #

@app.command()
def route(
    origin: Optional[str] = typer.Option(None, "--origin", "-o", help="origin stop id."),
    dest: Optional[str] = typer.Option(None, "--dest", "-D", help="destination stop id."),
    lambda_risk: float = typer.Option(0.0, "--lambda", "-l",
                                      help="risk sensitivity in f* = g + h + lambda*R."),
    list_stops: int = typer.Option(0, "--list", help="just list N well-connected stops and exit."),
) -> None:
    """Plan a risk-aware route between two stops on the Prague network."""
    _require(config.OUT_STOPS_ENRICHED, "Build it first: transit-control run --to 3")
    from transit.routing import TransportGraph

    _echo("Loading network ...")
    tg = TransportGraph.from_enriched(config.OUT_STOPS_ENRICHED, verbose=False)

    if list_stops:
        for s in tg.named_stops(limit=list_stops):
            _echo(f"  {s['id']:>10}  P={s['p_overload']:.3f}  {s.get('name','')}")
        return

    if not origin or not dest:
        typer.secho("Provide --origin and --dest (or --list N to browse).",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    for label, sid in (("origin", origin), ("dest", dest)):
        if not tg.has_stop(sid):
            typer.secho(f"Unknown {label} stop id: {sid}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    r = tg.route(origin, dest, lambda_risk=lambda_risk)
    if not r["reachable"]:
        typer.secho("No route found (stops in different components?).",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _echo(f"lambda={lambda_risk}:  {r['n_stops']} stops, "
          f"{r['length_km']:.2f} km, mean P(overload)={r['r_overload']:.3f}, "
          f"total risk={r['total_risk']:.2f}")
    _echo("path: " + " -> ".join(r["path"]))


# --------------------------------------------------------------------------- #
# decide (control action)
# --------------------------------------------------------------------------- #

@app.command()
def decide(
    p: float = typer.Option(..., "--p", help="overload probability in [0, 1]."),
    free_units: bool = typer.Option(True, "--free-units/--no-free-units",
                                    help="are spare vehicles available?"),
) -> None:
    """Map an overload probability onto a control action (a0..a5)."""
    from transit.decision import DecisionEngine

    d = DecisionEngine().decide(p, free_units_available=free_units)
    _echo(str(d))
    _echo(f"actions: {d.action_codes}   total relative cost: {d.total_cost}")


# --------------------------------------------------------------------------- #
# serve (Streamlit dashboard)
# --------------------------------------------------------------------------- #

@app.command()
def serve(
    port: int = typer.Option(8501, "--port", "-p", help="port for the dashboard."),
    headless: bool = typer.Option(False, "--headless", help="don't auto-open a browser."),
) -> None:
    """Launch the interactive Streamlit dashboard."""
    dashboard = Path(__file__).resolve().parent / "app" / "dashboard.py"
    _require(dashboard, "dashboard module not found in transit/app/.")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard),
           "--server.port", str(port)]
    if headless:
        cmd += ["--server.headless", "true"]
    _echo(f"$ {' '.join(cmd)}")
    try:
        raise typer.Exit(code=subprocess.run(cmd, cwd=config.BASE_DIR).returncode)
    except FileNotFoundError:
        typer.secho("Streamlit is not installed. Run: pip install -r requirements-app.txt",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (see pyproject ``[project.scripts]``)."""
    app()


if __name__ == "__main__":
    main()
