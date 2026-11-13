"""Protocolo de remuestreo del Hito II.

Semillas declaradas antes de ejecutar: 1..10 (la 42 ya consultó su test en la
primera entrega y se excluye). Para cada semilla se repite TODO el pipeline de
selección: partición 80/20, búsqueda RBF multiclase en dos etapas, selección de
la pareja binaria por OOF, búsqueda de kernels binarios (E2) y comparación
OvR/OvO a configuración común (E3). Cada test se consulta una sola vez.
"""

from pathlib import Path
import json
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split

import src.experiments as ex

SEEDS = list(range(1, 11))


def one_seed(seed, X, y):
    ex.SEED = seed  # _cv() y las particiones leen esta global
    t0 = perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    search, _ = ex._staged_rbf_search(X_train, y_train)
    C = float(search.best_params_["svc__C"])
    gamma = float(search.best_params_["svc__gamma"])
    oof = cross_val_predict(ex._pipeline("rbf", C=C, gamma=gamma), X_train, y_train, cv=ex._cv(), n_jobs=-1)
    pair, _, ranking = ex._pick_binary_pair(y_train, oof)

    row = {
        "seed": seed,
        "C": C,
        "gamma": gamma,
        "cv_f1_macro": float(search.best_score_),
        "pair": "-".join(pair),
        "pair_oof_confusions": int(ranking.iloc[0]["confusiones_OOF"]),
    }
    # E2 sobre la pareja elegida en esta semilla
    trm, tem = y_train.isin(pair), y_test.isin(pair)
    for name, est, grid in [
        ("lineal", ex._pipeline("linear"), {"svc__C": ex.C_VALUES}),
        (
            "polinomial",
            ex._pipeline("poly", coef0=1.0),
            {"svc__C": ex.C_VALUES, "svc__degree": [2, 3], "svc__gamma": ["scale"]},
        ),
    ]:
        s = ex._grid_search(X_train[trm], y_train[trm], est, grid)
        row[f"e2_{name}_test_f1"] = f1_score(y_test[tem], s.predict(X_test[tem]), average="macro")
    s, _ = ex._staged_rbf_search(X_train[trm], y_train[trm])
    row["e2_rbf_test_f1"] = f1_score(y_test[tem], s.predict(X_test[tem]), average="macro")

    # E3 a configuración común
    preds = {}
    for strategy in ["OvR", "OvO"]:
        model = ex._explicit_model(strategy, C, gamma).fit(X_train, y_train)
        preds[strategy] = model.predict(X_test)
        row[f"e3_{strategy}_f1_macro"] = f1_score(y_test, preds[strategy], average="macro")
        row[f"e3_{strategy}_accuracy"] = accuracy_score(y_test, preds[strategy])
    boot = ex._paired_bootstrap(y_test, preds["OvR"], preds["OvO"], seed=seed)
    row["e3_diff_ovo_minus_ovr"] = boot["observed_diff"]
    row["e3_boot_ci_low"] = boot["ci95_low"]
    row["e3_boot_ci_high"] = boot["ci95_high"]
    row["runtime_s"] = perf_counter() - t0
    return row


def main(root):
    root = Path(root)
    raw = pd.read_excel(root / "data" / "Dry_Bean_Dataset.xlsx")
    df = raw.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns="Class"), df["Class"]
    out = root / "results" / "hito2_remuestreo_semillas.csv"
    rows = []
    for seed in SEEDS:
        rows.append(one_seed(seed, X, y))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(json.dumps(rows[-1]), flush=True)
    res = pd.DataFrame(rows)
    num = res.select_dtypes("number").drop(columns=["seed"])
    summary = pd.DataFrame({"media": num.mean(), "de": num.std(ddof=1), "min": num.min(), "max": num.max()})
    summary.to_csv(root / "results" / "hito2_remuestreo_resumen.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
