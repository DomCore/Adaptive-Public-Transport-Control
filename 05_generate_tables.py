"""
05_generate_tables.py
=====================

Форматує результати у Markdown-таблиці для копі-пасту в дисертацію
(розділ 3.4.5).

Вхід:  output/results_aggregated.csv
       output/statistical_test.csv
Вихід: output/tables_for_dissertation.md
"""

import pandas as pd
from config import OUT_RESULTS_AGG, OUT_STATS, OUT_TABLES_MD, ALPHA


def fmt_table_3_6(df_agg: pd.DataFrame) -> str:
    """Таблиця 3.9: метрики маршруту по λ."""
    lines = [
        "## Таблиця 3.9 — Залежність метрик маршруту від параметра λ",
        "",
        "| λ    | R_overload | Δ довжини, % | Coverage | n_stops |",
        "|------|------------|--------------|----------|---------|",
    ]
    for _, r in df_agg.iterrows():
        d_len = r.get("delta_length_pct", float("nan"))
        d_len_str = f"{d_len:+.2f}" if pd.notna(d_len) else "—"
        lines.append(
            f"| {r['lambda']:.1f}  "
            f"| {r['avg_r_overload']:.4f}    "
            f"| {d_len_str:>12s} "
            f"| {r['coverage']:.3f}   "
            f"| {r['avg_n_stops']:.1f}    |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt_table_stats(df_stats: pd.DataFrame, alpha: float) -> str:
    """Парний t-тест."""
    lines = [
        "## Таблиця 3.10 — Парний t-тест: значущість зменшення R_overload",
        "",
        "| λ    | n   | Mean(λ=0) | Mean(λ) | Reduction | t-stat | p-value | Sig. (α=0.05) |",
        "|------|-----|-----------|---------|-----------|--------|---------|---------------|",
    ]
    for _, r in df_stats.iterrows():
        if "warning" in r and isinstance(r.get("warning"), str):
            lines.append(
                f"| {r['lambda']:.1f}  | {r['n_pairs']} "
                f"| —         | —       | —         | —      | —       | (skipped: too few pairs) |"
            )
            continue
        sig = "**ТАК**" if r["p_value"] < alpha else "ні"
        p_str = f"{r['p_value']:.2e}" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['lambda']:.1f}  "
            f"| {int(r['n_pairs'])} "
            f"| {r['mean_baseline']:.4f}    "
            f"| {r['mean_treated']:.4f}  "
            f"| {r['reduction_pct']:+.1f}%   "
            f"| {r['t_stat']:>6.2f} "
            f"| {p_str:>7s} "
            f"| {sig}      |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt_dissertation_text(df_agg: pd.DataFrame, df_stats: pd.DataFrame) -> str:
    """Готовий шаблон тексту для розділу 3.4.5 з підставленими цифрами."""
    if len(df_agg) == 0:
        return ""

    base_row = df_agg[df_agg["lambda"] == 0.0]
    high_row = df_agg[df_agg["lambda"] == df_agg["lambda"].max()]

    base_r = base_row["avg_r_overload"].iloc[0] if len(base_row) else float("nan")
    high_r = high_row["avg_r_overload"].iloc[0] if len(high_row) else float("nan")
    high_lam = high_row["lambda"].iloc[0] if len(high_row) else float("nan")
    high_dlen = high_row["delta_length_pct"].iloc[0] if len(high_row) else float("nan")
    high_cov = high_row["coverage"].iloc[0] if len(high_row) else float("nan")

    # Шукаємо λ ≈ 2 для "точки балансу"
    if 2.0 in df_agg["lambda"].values:
        balance = df_agg[df_agg["lambda"] == 2.0].iloc[0]
        bal_r = balance["avg_r_overload"]
        bal_dlen = balance["delta_length_pct"]
        bal_cov = balance["coverage"]
    else:
        bal_r = bal_dlen = bal_cov = float("nan")

    sig_lambdas = df_stats[
        df_stats.get("significant_at_alpha", False) == True
    ]["lambda"].tolist() if "significant_at_alpha" in df_stats.columns else []

    text = f"""## Готовий шаблон для розділу 3.4.5

**Інтерпретація результатів.** Зі зростанням λ спостерігається монотонне 
зменшення R_overload (з {base_r:.3f} при λ=0 до {high_r:.3f} при λ={high_lam:.0f}, 
тобто зменшення на {(base_r - high_r) / base_r * 100:.1f}%) ціною збільшення 
довжини маршруту (на {high_dlen:.1f}% при λ={high_lam:.0f}). Покриття зупинок 
зменшується з 100% до {high_cov*100:.1f}%, оскільки при високих λ окремі зупинки 
стають недосяжними через високу ймовірність перевантаження сусідів.

**Статистична значущість.** Парний t-тест між R_overload при λ=0 і λ>0 
дав p < {ALPHA} для {len(sig_lambdas)} з {len(df_stats)} тестованих 
значень λ, що підтверджує статистичну значущість зменшення ризику.

**Точка балансу.** Експертне налаштування системи відповідає λ ≈ 2, 
що забезпечує зменшення ризику з {base_r:.3f} до {bal_r:.3f} 
({(base_r - bal_r) / base_r * 100:.1f}% зниження) ціною збільшення 
довжини на {bal_dlen:.1f}% при збереженні покриття {bal_cov*100:.1f}%.
"""
    return text


def main():
    df_agg = pd.read_csv(OUT_RESULTS_AGG)
    df_stats = pd.read_csv(OUT_STATS)

    parts = [
        "# Таблиці для дисертації — підрозділ 3.4.5",
        "",
        "Згенеровано автоматично скриптом `05_generate_tables.py`.",
        "",
        fmt_table_3_6(df_agg),
        fmt_table_stats(df_stats, ALPHA),
        fmt_dissertation_text(df_agg, df_stats),
    ]
    out = "\n".join(parts)

    with open(OUT_TABLES_MD, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[05] Таблиці збережено: {OUT_TABLES_MD}")
    print("\n" + out)


if __name__ == "__main__":
    main()
