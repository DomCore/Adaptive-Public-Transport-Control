"""
01_prepare_data.py
==================

Препроцесинг NY Bus Breakdown + створення нових вузлів для розширеної BBN:
  - overload    (бінарний; проксі = подія Breakdown)
  - TimeOfDay   (категоріальний; з Occurred_On)
  - DayOfWeek   (weekday/weekend; з Occurred_On)

Вхід:  data/data.csv
Вихід: output/data_extended.parquet
"""

import re
import pandas as pd
import numpy as np

from config import (
    INPUT_CSV,
    OUT_DATA_EXTENDED,
    TIME_OF_DAY_BINS,
)


# ---------- Helpers ----------

def parse_delay_minutes(val) -> float:
    """
    Парсить тривалість затримки з рядка у хвилини.

    Формати, що підтримуються:
      '15 min', '30 min'             → 15, 30
      '1 hour', '2 hours'            → 60, 120
      '1 hour 30 min', '1.5 hour'    → 90
      '15-30', '15 to 30', '15/30'   → середнє 22.5
      'over 30', 'more than 30'      → 30
      '90', '90.0'                   → 90

    Повертає 0 для NaN / порожніх значень (ВАЖЛИВО: не NaN!),
    бо в датасеті порожнє How_Long_Delayed = 'не було затримки'.
    """
    if pd.isnull(val):
        return 0.0
    s = str(val).lower().strip()
    if not s or s in ('nan', 'none', 'unknown', '0', '0 min', 'no delay'):
        return 0.0

    # Діапазон: "15-30" або "15 to 30"
    range_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|/)\s*(\d+(?:\.\d+)?)', s)
    if range_match:
        a, b = float(range_match.group(1)), float(range_match.group(2))
        return (a + b) / 2

    # Години (в т.ч. дробові: "1.5 hour")
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hour|hr|год)', s)
    h = float(hours_match.group(1)) if hours_match else 0.0

    # Хвилини
    minutes_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:min|хв)', s)
    m = float(minutes_match.group(1)) if minutes_match else 0.0

    if h or m:
        return h * 60 + m

    # Останній шанс: просте число → хвилини
    digit_match = re.search(r'(\d+(?:\.\d+)?)', s)
    if digit_match:
        return float(digit_match.group(1))

    return 0.0


def assign_time_of_day(hour: int) -> str:
    """Година (0–23) → категорія часу доби."""
    for start, end, label in TIME_OF_DAY_BINS:
        if start <= hour < end:
            return label
    return "evening"


# ---------- Pipeline ----------

def load_and_clean(path) -> pd.DataFrame:
    print(f"[01] Завантаження {path} ...")
    cols = [
        "Boro",
        "Reason",
        "Breakdown_or_Running_Late",
        "How_Long_Delayed",
        "Occurred_On",
        "School_Age_or_PreK",
        "Has_Contractor_Notified_Schools",
        "Number_Of_Students_On_The_Bus",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    print(f"     Записів: {len(df):,}")
    print(f"     Колонок:  {list(df.columns)}")
    return df


def add_overload_node(df: pd.DataFrame) -> pd.DataFrame:
    """
    Бінарна цільова змінна 'overload'.

    У датасеті NY Bus поле How_Long_Delayed заповнене лише в ~0.1% записів,
    тому пороговий підхід (delay > N min) дає вироджений розподіл.

    Натомість використовуємо Breakdown_or_Running_Late == 'Breakdown'
    як проксі для перевантаження: поломка автобуса прямо вказує на
    серйозний збій у обслуговуванні маршруту.
    """
    print("[01] Додавання вузла 'overload' (= Breakdown_or_Running_Late == 'Breakdown') ...")
    df["overload"] = (
            df["Breakdown_or_Running_Late"].astype(str).str.strip() == "Breakdown"
    ).astype(int)
    rate = df["overload"].mean()
    print(f"     Частка overload=1: {rate:.1%} (Breakdown events)")
    return df


def add_temporal_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """TimeOfDay і DayOfWeek з Occurred_On."""
    print("[01] Додавання вузлів 'TimeOfDay' і 'DayOfWeek' ...")
    df["Occurred_On"] = pd.to_datetime(df["Occurred_On"], errors="coerce")
    parsed = df["Occurred_On"].notna().sum()
    print(f"     Розпарсено дат: {parsed:,} з {len(df):,}")

    if parsed < len(df) * 0.5:
        # Fallback: рівномірний розподіл годин
        print("     [!] Менше 50% валідних дат — використовується синтетичний розподіл")
        rng = np.random.default_rng(42)
        df["hour"] = rng.integers(0, 24, size=len(df))
        df["weekday"] = rng.integers(0, 7, size=len(df))
    else:
        df["hour"] = df["Occurred_On"].dt.hour.fillna(12).astype(int)
        df["weekday"] = df["Occurred_On"].dt.dayofweek.fillna(0).astype(int)

    df["TimeOfDay"] = df["hour"].apply(assign_time_of_day)
    df["DayOfWeek"] = np.where(df["weekday"] < 5, "weekday", "weekend")

    print(f"     TimeOfDay розподіл:\n{df['TimeOfDay'].value_counts().to_string()}")
    print(f"     DayOfWeek розподіл:\n{df['DayOfWeek'].value_counts().to_string()}")
    return df


def add_delay_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дискретизує тривалість затримки у категоріальний вузол How_Long_Delayed.

    Сире текстове поле How_Long_Delayed парситься у хвилини
    (parse_delay_minutes) і бінується у 4 категорії. Цей вузол є нащадком
    Reason у БМД (а не предком overload), тому НЕ впливає на маршрутизацію —
    додається для відповідності концептуальній структурі (підрозділ 2.1.2).
    """
    print("[01] Додавання вузла 'How_Long_Delayed' (дискретизація затримки) ...")
    minutes = df["How_Long_Delayed"].apply(parse_delay_minutes)

    def to_bin(m: float) -> str:
        if m <= 0:
            return "none"      # порожнє поле = не було тривалої затримки
        if m <= 15:
            return "short"     # 0–15 хв
        if m <= 40:
            return "medium"    # 15–40 хв
        return "long"          # 40+ хв

    df["How_Long_Delayed"] = minutes.apply(to_bin)
    print(f"     How_Long_Delayed розподіл:\n"
          f"{df['How_Long_Delayed'].value_counts().to_string()}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Заповнюємо NaN значенням 'Unknown' і приводимо до str для pgmpy."""
    print("[01] Кодування категоріальних ...")
    cat_cols = ["Boro", "Reason", "Breakdown_or_Running_Late",
                "TimeOfDay", "DayOfWeek", "School_Age_or_PreK",
                "Has_Contractor_Notified_Schools", "How_Long_Delayed"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Залишаємо тільки 8 вузлів BBN + допоміжний Breakdown_or_Running_Late."""
    keep = [
        "TimeOfDay", "DayOfWeek", "Boro", "School_Age_or_PreK",
        "Reason", "overload", "How_Long_Delayed",
        "Has_Contractor_Notified_Schools",
        "Breakdown_or_Running_Late",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def main():
    df = load_and_clean(INPUT_CSV)
    df = add_overload_node(df)
    df = add_temporal_nodes(df)
    df = add_delay_category(df)
    df = encode_categorical(df)
    df = select_final_columns(df)

    # Видаляємо рядки з пропусками у ключових колонках ядра BBN.
    # ВАЖЛИВО: набір умов фільтрації не змінюється при розширенні мережі —
    # це гарантує однаковий набір рядків і, як наслідок, незмінні CPD вузлів
    # overload та Reason (а отже — незмінні результати маршрутизації).
    before = len(df)
    df = df.dropna(subset=["overload", "TimeOfDay", "Reason", "Breakdown_or_Running_Late"])
    df = df[df["Reason"] != "Unknown"]
    df = df[df["Breakdown_or_Running_Late"] != "Unknown"]
    print(f"     Після фільтрів: Reason≠Unknown, BoRL≠Unknown")
    print(f"[01] Після dropna: {len(df):,} (видалено {before - len(df):,})")

    # Зберігаємо
    df.to_parquet(OUT_DATA_EXTENDED, index=False)
    print(f"[01] Збережено: {OUT_DATA_EXTENDED}")

    # Summary
    print("\n=== Зведення ===")
    print(f"Записів у датасеті:       {len(df):,}")
    print(f"Унікальні Boro:           {df['Boro'].nunique()}")
    print(f"Унікальні Reason:         {df['Reason'].nunique()}")
    print(f"Частка overload=1:        {df['overload'].mean():.1%}")
    print(f"\nBoro × overload (умовна частота overload=1 по районах):")
    print(df.groupby("Boro")["overload"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
