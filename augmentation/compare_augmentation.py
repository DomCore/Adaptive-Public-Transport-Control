"""
compare_augmentation.py
=======================

Порівняння методів аугментації для NY Bus Breakdown.
ВАРІАНТ 2: класифікація Breakdown vs Running Late на контекстних ознаках.

  - target = (Breakdown_or_Running_Late == 'Breakdown')  — клас 1
  - features = ТІЛЬКИ контекстні: n_students + one-hot для Boro, Reason,
    TimeOfDay, DayOfWeek
  - БЕЗ delay_min/How_Long_Delayed (їх взагалі викинуто)

Методи:
  - Baseline (без аугментації)
  - Random Oversampling
  - SMOTE (власна реалізація)
  - ADASYN (власна реалізація)
  - GAN (як у model.py дисертації, але адаптований до високої розмірності)

Вхід:  ../data/data.csv
Вихід: output/comparison_results.csv
       output/comparison_table.md

Потребує TensorFlow — встанови augmentation/requirements-gan.txt поверх
кореневого requirements.txt.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ---------- Шляхи ----------

BASE = Path(__file__).parent
# data.csv lives in the repo-level data/ directory (see data/README.md).
INPUT_CSV = BASE.parent / "data" / "data.csv"

OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_RESULTS = OUTPUT_DIR / "comparison_results.csv"
OUT_TABLE_MD = OUTPUT_DIR / "comparison_table.md"


# ---------- 1. Завантаження + препроцесинг ----------

TIME_OF_DAY_BINS = [
    (0, 6, "night"),
    (6, 10, "peak_morning"),
    (10, 16, "midday"),
    (16, 19, "peak_evening"),
    (19, 24, "evening"),
]


def assign_time_of_day(hour: int) -> str:
    for s, e, label in TIME_OF_DAY_BINS:
        if s <= hour < e:
            return label
    return "evening"


def load_data() -> tuple[np.ndarray, np.ndarray, list]:
    """
    Готує (X, y, feature_names).
    X — числові + one-hot, без delay-related ознак.
    y — Breakdown (1) vs Running Late (0).

    Викидаємо записи з Breakdown_or_Running_Late = Unknown — вони не дають
    сигналу і змішують класи.
    """
    print(f"[load] Читаю {INPUT_CSV} ...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Не знайдено {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False, usecols=lambda c: c in [
        "Number_Of_Students_On_The_Bus",
        "Breakdown_or_Running_Late",
        "Boro",
        "Reason",
        "Occurred_On",
    ])
    print(f"       Записів: {len(df):,}")

    # Цільова: Breakdown vs Running Late
    df["BoRL"] = df["Breakdown_or_Running_Late"].astype(str).str.strip()
    df = df[df["BoRL"].isin(["Breakdown", "Running Late"])].copy()
    df["target"] = (df["BoRL"] == "Breakdown").astype(int)
    print(f"       Після фільтра BoRL: {len(df):,}")

    # n_students числова
    df["n_students"] = pd.to_numeric(
        df["Number_Of_Students_On_The_Bus"], errors="coerce"
    ).fillna(0).clip(lower=0, upper=100)

    # TimeOfDay і DayOfWeek з Occurred_On
    df["Occurred_On"] = pd.to_datetime(df["Occurred_On"], errors="coerce")
    parsed = df["Occurred_On"].notna().sum()
    print(f"       Розпарсено дат: {parsed:,} з {len(df):,}")
    if parsed < len(df) * 0.5:
        rng = np.random.default_rng(42)
        df["hour"] = rng.integers(0, 24, size=len(df))
        df["weekday"] = rng.integers(0, 7, size=len(df))
    else:
        df["hour"] = df["Occurred_On"].dt.hour.fillna(12).astype(int)
        df["weekday"] = df["Occurred_On"].dt.dayofweek.fillna(0).astype(int)
    df["TimeOfDay"] = df["hour"].apply(assign_time_of_day)
    df["DayOfWeek"] = np.where(df["weekday"] < 5, "weekday", "weekend")

    # Категоріальні
    for col in ["Boro", "Reason"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    print(f"       Розподіл target: 0=Running Late ({sum(df['target']==0):,}), "
          f"1=Breakdown ({sum(df['target']==1):,}, "
          f"{df['target'].mean()*100:.1f}%)")

    # One-hot
    cat_cols = ["Boro", "Reason", "TimeOfDay", "DayOfWeek"]
    df_onehot = pd.get_dummies(df[cat_cols], columns=cat_cols, drop_first=False)

    # Збираємо матрицю ознак
    X_df = pd.concat([df[["n_students"]], df_onehot], axis=1)
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)
    y = df["target"].values

    print(f"       Ознак: {len(feature_names)} (n_students + one-hot)")
    return X, y, feature_names


# ---------- 2. SMOTE/ADASYN/RandomOversample (pure NumPy) ----------

def random_oversample(X, y, target_ratio=0.3, random_state=42):
    rng = np.random.default_rng(random_state)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_target = int(len(X_maj) * target_ratio)
    n_synth = max(0, n_target - len(X_min))
    if n_synth == 0:
        return X, y
    idx = rng.integers(0, len(X_min), n_synth)
    return np.vstack([X, X_min[idx]]), np.concatenate([y, np.ones(n_synth, dtype=int)])


def smote_oversample(X, y, target_ratio=0.3, k_neighbors=5, random_state=42):
    rng = np.random.default_rng(random_state)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    if len(X_min) < 2:
        return X, y
    n_target = int(len(X_maj) * target_ratio)
    n_synth = max(0, n_target - len(X_min))
    if n_synth == 0:
        return X, y

    k = min(k_neighbors, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_min)
    _, neighbors = nn.kneighbors(X_min)
    neighbors = neighbors[:, 1:]

    base_idx = rng.integers(0, len(X_min), n_synth)
    nbr_local_idx = rng.integers(0, k, n_synth)
    nbr_idx = neighbors[base_idx, nbr_local_idx]
    alpha = rng.random((n_synth, 1))

    X_synth = X_min[base_idx] + alpha * (X_min[nbr_idx] - X_min[base_idx])
    return np.vstack([X, X_synth]), np.concatenate([y, np.ones(n_synth, dtype=int)])


def adasyn_oversample(X, y, target_ratio=0.3, k_neighbors=5, random_state=42):
    rng = np.random.default_rng(random_state)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    if len(X_min) < 2:
        return X, y
    n_target = int(len(X_maj) * target_ratio)
    n_synth = max(0, n_target - len(X_min))
    if n_synth == 0:
        return X, y

    k = min(k_neighbors, len(X) - 1)
    nn_all = NearestNeighbors(n_neighbors=k + 1)
    nn_all.fit(X)
    _, idx_all = nn_all.kneighbors(X_min)
    idx_all = idx_all[:, 1:]

    r = (y[idx_all] == 0).sum(axis=1) / k
    if r.sum() == 0:
        return smote_oversample(X, y, target_ratio, k_neighbors, random_state)
    r_norm = r / r.sum()
    n_per_minority = np.round(r_norm * n_synth).astype(int)

    k_min = min(k_neighbors, len(X_min) - 1)
    nn_min = NearestNeighbors(n_neighbors=k_min + 1)
    nn_min.fit(X_min)
    _, neighbors_min = nn_min.kneighbors(X_min)
    neighbors_min = neighbors_min[:, 1:]

    synthetic_list = []
    for i, n_i in enumerate(n_per_minority):
        if n_i == 0: continue
        nbr_local = rng.integers(0, k_min, n_i)
        nbr_idx = neighbors_min[i, nbr_local]
        alpha = rng.random((n_i, 1))
        synth_i = X_min[i] + alpha * (X_min[nbr_idx] - X_min[i])
        synthetic_list.append(synth_i)

    if not synthetic_list:
        return X, y
    X_synth = np.vstack(synthetic_list)
    return np.vstack([X, X_synth]), np.concatenate([y, np.ones(len(X_synth), dtype=int)])


# ---------- 3. GAN — адаптований до high-dim ----------

def gan_oversample(X_train, y_train, target_ratio=0.3,
                   latent_dim=16, epochs=50, batch_size=512, random_state=42):
    """
    GAN для high-dim даних.
    
    Архітектура побільшена порівняно з model.py (бо там було 2 ознаки,
    а тут ~30):
      Generator: Dense(64, ReLU) → Dense(128, ReLU) → Dense(n_features, Linear)
      Discriminator: Dense(128, LeakyReLU) → Dropout(0.3) → Dense(64, LeakyReLU)
                     → Dense(1, Sigmoid)
    
    Це стандартна табличного GAN для ~30 ознак. Архітектура з model.py
    (16/32 нейрони) занадто мала для 30 features.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    X_min = X_train[y_train == 1]
    if len(X_min) < 100:
        print(f"       [!] Замало мінорних ({len(X_min)}) для GAN")
        return X_train, y_train

    n_features = X_min.shape[1]

    # Generator
    g_input = layers.Input(shape=(latent_dim,))
    h = layers.Dense(64, activation="relu")(g_input)
    h = layers.Dense(128, activation="relu")(h)
    g_output = layers.Dense(n_features, activation="linear")(h)
    generator = models.Model(g_input, g_output)

    # Discriminator
    d_input = layers.Input(shape=(n_features,))
    h = layers.Dense(128)(d_input)
    h = layers.LeakyReLU(alpha=0.2)(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(64)(h)
    h = layers.LeakyReLU(alpha=0.2)(h)
    d_output = layers.Dense(1, activation="sigmoid")(h)
    discriminator = models.Model(d_input, d_output)
    discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                          loss="binary_crossentropy", metrics=["accuracy"])

    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                loss="binary_crossentropy")

    print(f"       GAN training ({epochs} epochs, {len(X_min)} minority samples, "
          f"{n_features} features)...")
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, len(X_min), batch_size)
        real_batch = X_min[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_batch = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_batch, fake_labels)
        discriminator.trainable = False

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        if (epoch + 1) % 10 == 0:
            d_val = float(d_loss_real[0] if hasattr(d_loss_real, '__len__') else d_loss_real)
            g_val = float(g_loss if not hasattr(g_loss, '__len__') else g_loss[0])
            print(f"         epoch {epoch+1:3d}: d_loss={d_val:.3f}, g_loss={g_val:.3f}")

    n_maj = sum(y_train == 0)
    n_target = int(n_maj * target_ratio)
    n_synth = max(0, n_target - len(X_min))
    if n_synth == 0:
        return X_train, y_train

    noise = np.random.normal(0, 1, (n_synth, latent_dim))
    X_synth = generator.predict(noise, verbose=0)

    print(f"       GAN додав: +{n_synth} synthetic minority samples")
    return np.vstack([X_train, X_synth]), np.concatenate([y_train, np.ones(n_synth, dtype=int)])


# ---------- 4. Оцінка ----------

def train_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """RF + LR на зразок 3.4.2 дисертації, плюс ROC-AUC."""
    results = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                random_state=42, n_jobs=-1,
                                class_weight=None)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    results["rf_acc"] = accuracy_score(y_test, y_pred)
    results["rf_prec"] = precision_score(y_test, y_pred, zero_division=0)
    results["rf_rec"] = recall_score(y_test, y_pred, zero_division=0)
    results["rf_f1"] = f1_score(y_test, y_pred, zero_division=0)
    results["rf_auc"] = roc_auc_score(y_test, y_proba)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs",
                            C=1.0, class_weight=None)
    lr.fit(X_train_sc, y_train)
    y_pred = lr.predict(X_test_sc)
    y_proba = lr.predict_proba(X_test_sc)[:, 1]
    results["lr_acc"] = accuracy_score(y_test, y_pred)
    results["lr_prec"] = precision_score(y_test, y_pred, zero_division=0)
    results["lr_rec"] = recall_score(y_test, y_pred, zero_division=0)
    results["lr_f1"] = f1_score(y_test, y_pred, zero_division=0)
    results["lr_auc"] = roc_auc_score(y_test, y_proba)

    return results


# ---------- 5. Main ----------

def main():
    X, y, feat_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"\n[split] Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"        Train balance: 0={sum(y_train==0):,}, 1={sum(y_train==1):,}")
    print(f"        Test  balance: 0={sum(y_test==0):,}, 1={sum(y_test==1):,}\n")

    methods = [
        ("Baseline", lambda X, y: (X, y)),
        ("Random Oversampling", random_oversample),
        ("SMOTE", smote_oversample),
        ("ADASYN", adasyn_oversample),
        ("GAN", gan_oversample),
    ]

    rows = []
    for name, fn in methods:
        print(f"=== {name} ===")
        try:
            X_aug, y_aug = fn(X_train, y_train)
            print(f"       Після аугментації: train={len(X_aug):,}, "
                  f"minority={sum(y_aug==1):,}")
            metrics = train_and_evaluate(X_aug, y_aug, X_test, y_test)
            row = {"method": name, "n_train": len(X_aug),
                   "n_minority": int(sum(y_aug == 1)), **metrics}
            rows.append(row)
            print(f"       RF: acc={metrics['rf_acc']:.3f} "
                  f"prec={metrics['rf_prec']:.3f} "
                  f"rec={metrics['rf_rec']:.3f} "
                  f"f1={metrics['rf_f1']:.3f} "
                  f"auc={metrics['rf_auc']:.3f}")
            print(f"       LR: acc={metrics['lr_acc']:.3f} "
                  f"prec={metrics['lr_prec']:.3f} "
                  f"rec={metrics['lr_rec']:.3f} "
                  f"f1={metrics['lr_f1']:.3f} "
                  f"auc={metrics['lr_auc']:.3f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"method": name, "error": str(e)})
        print()

    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUT_RESULTS, index=False)
    print(f"[save] Сирі результати: {OUT_RESULTS}")

    # Markdown
    md = ["# Порівняння методів аугментації (Варіант 2 — без leakage)\n",
          f"**Цільова змінна:** Breakdown (n={sum(y==1):,}) vs Running Late (n={sum(y==0):,})\n",
          f"**Ознаки:** {len(feat_names)} (n_students + one-hot для Boro, Reason, TimeOfDay, DayOfWeek)\n",
          "## Random Forest\n",
          "| Метод | Accuracy | Precision | Recall | F1 | ROC-AUC |",
          "|-------|----------|-----------|--------|-----|---------|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['method']} | ERROR: {r['error']} | | | | |")
            continue
        md.append(f"| {r['method']} | {r['rf_acc']:.3f} | {r['rf_prec']:.3f} "
                  f"| {r['rf_rec']:.3f} | {r['rf_f1']:.3f} | {r['rf_auc']:.3f} |")

    md += ["", "## Logistic Regression\n",
           "| Метод | Accuracy | Precision | Recall | F1 | ROC-AUC |",
           "|-------|----------|-----------|--------|-----|---------|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['method']} | ERROR | | | | |")
            continue
        md.append(f"| {r['method']} | {r['lr_acc']:.3f} | {r['lr_prec']:.3f} "
                  f"| {r['lr_rec']:.3f} | {r['lr_f1']:.3f} | {r['lr_auc']:.3f} |")

    OUT_TABLE_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[save] Markdown таблиця: {OUT_TABLE_MD}\n")

    print("\n=== ПІДСУМКОВА ТАБЛИЦЯ ===\n")
    print("Random Forest:")
    print(f"{'Method':<22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    for r in rows:
        if "error" in r: continue
        print(f"{r['method']:<22s} {r['rf_acc']:>6.3f} {r['rf_prec']:>6.3f} "
              f"{r['rf_rec']:>6.3f} {r['rf_f1']:>6.3f} {r['rf_auc']:>6.3f}")
    print("\nLogistic Regression:")
    print(f"{'Method':<22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    for r in rows:
        if "error" in r: continue
        print(f"{r['method']:<22s} {r['lr_acc']:>6.3f} {r['lr_prec']:>6.3f} "
              f"{r['lr_rec']:>6.3f} {r['lr_f1']:>6.3f} {r['lr_auc']:>6.3f}")


if __name__ == "__main__":
    main()
