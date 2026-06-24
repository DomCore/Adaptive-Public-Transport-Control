"""
02_build_bbn_pgmpy.py
=====================

Будуємо multi-parent Bayesian Network через pgmpy:
  - Підгонка CPD методом MaximumLikelihoodEstimator
  - Validation моделі
  - Тест Variable Elimination на прикладі
  - Збереження моделі через pickle

Вхід:  output/data_extended.parquet
Вихід: output/bbn_model.pkl
       output/bbn_meta.json
"""

import json
import pickle
import pandas as pd

# Сумісність: BayesianNetwork — назва у pgmpy <= 0.1.25;
# у нових версіях (>= 0.1.26) клас перейменовано на DiscreteBayesianNetwork.
try:
    from pgmpy.models import BayesianNetwork
except ImportError:  # pragma: no cover — залежить від версії pgmpy
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from config import (
    OUT_DATA_EXTENDED,
    OUT_BBN_MODEL,
    OUT_BBN_META,
    BBN_NODES,
    BBN_EDGES,
)


def build_model(df: pd.DataFrame) -> BayesianNetwork:
    """Будує BBN зі структурою з config і навчає CPD на даних."""
    print("[02] Створення BayesianNetwork зі структурою ...")
    print(f"     Вузли:  {BBN_NODES}")
    print(f"     Ребра:  {BBN_EDGES}")

    # Перевіряємо що всі вузли є в даних
    missing = [n for n in BBN_NODES if n not in df.columns]
    if missing:
        raise ValueError(f"Вузли відсутні в даних: {missing}")

    model = BayesianNetwork(BBN_EDGES)

    print("[02] Підгонка CPD методом Maximum Likelihood ...")
    model.fit(
        df[BBN_NODES],
        estimator=MaximumLikelihoodEstimator,
    )
    return model


def validate(model: BayesianNetwork) -> dict:
    """Перевірки моделі."""
    print("[02] Валідація моделі ...")
    is_valid = model.check_model()
    print(f"     check_model(): {is_valid}")

    summary = {
        "n_nodes": len(model.nodes()),
        "n_edges": len(model.edges()),
        "nodes": list(model.nodes()),
        "edges": [list(e) for e in model.edges()],
        "valid": bool(is_valid),
    }

    # Покажемо по кожному CPD: вузол, кількість batches, кардинальність
    print("\n     CPD overview:")
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is not None:
            n_states = cpd.variable_card
            n_evidence = len(cpd.variables) - 1
            print(f"       {node:30s} states={n_states:3d}  parents={n_evidence}")

    return summary


def smoke_test_inference(model: BayesianNetwork):
    """Запускаємо одну query, щоб переконатися що інференс працює."""
    print("\n[02] Smoke-test: інференс P(overload | TimeOfDay='peak_morning') ...")
    inference = VariableElimination(model)

    # Беремо валідні значення TimeOfDay з CPD
    tod_cpd = model.get_cpds("TimeOfDay")
    if tod_cpd is None:
        # TimeOfDay — корінь, дивимось state_names
        for parent_node in ["TimeOfDay"]:
            states = model.get_cpds(parent_node).state_names[parent_node]
            print(f"     States {parent_node}: {states}")
            break

    states = model.get_cpds("TimeOfDay").state_names["TimeOfDay"]
    test_state = "peak_morning" if "peak_morning" in states else states[0]

    try:
        result = inference.query(
            variables=["overload"],
            evidence={"TimeOfDay": test_state},
            show_progress=False,
        )
        print(f"     Q: P(overload | TimeOfDay={test_state})")
        print(f"     A: {dict(zip(result.state_names['overload'], result.values))}")
        return True
    except Exception as e:
        print(f"     [!] Інференс впав: {e}")
        return False


def main():
    print(f"[02] Завантаження {OUT_DATA_EXTENDED} ...")
    df = pd.read_parquet(OUT_DATA_EXTENDED)
    print(f"     Записів: {len(df):,}")

    model = build_model(df)
    summary = validate(model)
    smoke_ok = smoke_test_inference(model)
    summary["smoke_test_passed"] = smoke_ok

    # Зберігаємо
    print(f"\n[02] Збереження моделі: {OUT_BBN_MODEL}")
    with open(OUT_BBN_MODEL, "wb") as f:
        pickle.dump(model, f)

    print(f"[02] Збереження метаінформації: {OUT_BBN_META}")
    with open(OUT_BBN_META, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Зведення ===")
    print(f"Вузлів:                {summary['n_nodes']}")
    print(f"Ребер:                 {summary['n_edges']}")
    print(f"Модель валідна:        {summary['valid']}")
    print(f"Smoke-test:            {'OK' if smoke_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
