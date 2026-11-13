"""Experimentos reproducibles de clasificación con SVM: E1--E5.

Decisiones metodológicas importantes:
- se eliminan duplicados exactos antes de la partición;
- no hay imputación porque no existen valores faltantes;
- el test 20 % queda retenido hasta la evaluación final;
- toda selección usa CV estratificada de cinco pliegues en entrenamiento;
- la pareja binaria se elige con confusiones OOF del entrenamiento, nunca con test.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .smo import SMOBinary, SMOOneVsOne, SMOOneVsRest

SEED = 42
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
# Rejilla unidimensional amplia para kernels lineal/polinomial.
C_VALUES = [float(2**e) for e in [-5, -1, 3, 7]]
COARSE_C_EXP = [-5, -1, 3, 7, 11]
COARSE_GAMMA_EXP = [-15, -11, -7, -3, 1]
BOOTSTRAP_REPS = 2000


def _environment(root):
    """Versiones exactas y hardware de la corrida (necesario para E5)."""
    import importlib.metadata as md
    import sklearn

    def _cmd(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, cwd=root, timeout=10).stdout.strip()
        except Exception:
            return ""

    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gb = None
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError, AttributeError):
        pass
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "openpyxl"]:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    return {
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "packages": packages,
        "sklearn_build": sklearn.__version__,
    }


def _fingerprint(root, extra):
    """Huella de datos + configuración + versiones para validar la caché."""
    import sklearn

    data_hash = hashlib.sha256((root / "data" / "Dry_Bean_Dataset.xlsx").read_bytes()).hexdigest()
    payload = {
        "data_sha256": data_hash,
        "seed": SEED,
        "coarse_C_exp": COARSE_C_EXP,
        "coarse_gamma_exp": COARSE_GAMMA_EXP,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        **extra,
    }
    return payload


def _cache_valid(meta_path, fingerprint):
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")) == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def _paired_bootstrap(y_true, pred_a, pred_b, reps=BOOTSTRAP_REPS, seed=SEED):
    """IC 95 % percentil de F1macro(b) - F1macro(a) remuestreando el test.

    Es pareado: cada réplica usa los mismos índices para ambas estrategias.
    No requiere reentrenar; mide la incertidumbre por muestreo del test,
    no la variabilidad del entrenamiento.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        diffs[r] = f1_score(y_true[idx], pred_b[idx], average="macro", labels=CLASSES, zero_division=0) - f1_score(
            y_true[idx], pred_a[idx], average="macro", labels=CLASSES, zero_division=0
        )
    observed = f1_score(y_true, pred_b, average="macro") - f1_score(y_true, pred_a, average="macro")
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    return {
        "observed_diff": float(observed),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "bootstrap_sd": float(diffs.std(ddof=1)),
        "share_diff_le_0": float(np.mean(diffs <= 0)),
        "reps": reps,
        "n_test": n,
        "only_a_correct": int(np.sum(a_ok & ~b_ok)),
        "only_b_correct": int(np.sum(~a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
        "identical_predictions": int(np.sum(pred_a == pred_b)),
    }


def _pipeline(kernel="rbf", **kwargs):
    # No se incluye SimpleImputer: el dataset verificado no tiene faltantes.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel=kernel, cache_size=1000, **kwargs)),
        ]
    )


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def _grid_search(X, y, estimator, grid):
    search = GridSearchCV(
        estimator,
        grid,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        cv=_cv(),
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X, y)
    return search


def _staged_rbf_search(X, y):
    """Búsqueda base 2 amplia y refinamiento local reproducible.

    La fase gruesa cubre cinco órdenes de magnitud. La fase fina añade los
    exponentes vecinos del mejor punto para reducir el costo de evaluar una
    malla cartesiana completa.
    """
    coarse = _grid_search(
        X,
        y,
        _pipeline("rbf"),
        {
            "svc__C": [float(2**e) for e in COARSE_C_EXP],
            "svc__gamma": [float(2**e) for e in COARSE_GAMMA_EXP],
        },
    )
    best_c_exp = int(round(np.log2(float(coarse.best_params_["svc__C"]))))
    best_g_exp = int(round(np.log2(float(coarse.best_params_["svc__gamma"]))))
    fine_c_exp = sorted({e for e in range(best_c_exp - 1, best_c_exp + 2) if -5 <= e <= 15})
    fine_g_exp = sorted({e for e in range(best_g_exp - 1, best_g_exp + 2) if -15 <= e <= 3})
    fine = _grid_search(
        X,
        y,
        _pipeline("rbf"),
        {
            "svc__C": [float(2**e) for e in fine_c_exp],
            "svc__gamma": [float(2**e) for e in fine_g_exp],
        },
    )
    chosen = fine if fine.best_score_ >= coarse.best_score_ else coarse
    coarse_df = pd.DataFrame(coarse.cv_results_).assign(search_phase="coarse")
    fine_df = pd.DataFrame(fine.cv_results_).assign(search_phase="refinement")
    combined = pd.concat([coarse_df, fine_df], ignore_index=True)
    return chosen, combined


def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _fit_and_measure(model, X_train, y_train, X_test, y_test):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_s = perf_counter() - start
    start = perf_counter()
    pred = model.predict(X_test)
    predict_s = perf_counter() - start
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "fit_s": float(fit_s),
        "predict_s": float(predict_s),
    }, pred


def _explicit_model(strategy, C, gamma):
    base = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=1000)
    wrapper = OneVsRestClassifier(base) if strategy == "OvR" else OneVsOneClassifier(base)
    return Pipeline([("scaler", StandardScaler()), ("multi", wrapper)])


def _support_count(model):
    return int(sum(len(est.support_) for est in model.named_steps["multi"].estimators_))


def _pick_binary_pair(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    rows = []
    for i, first in enumerate(CLASSES):
        for j in range(i + 1, len(CLASSES)):
            second = CLASSES[j]
            errors = int(cm[i, j] + cm[j, i])
            rows.append({"clase_1": first, "clase_2": second, "confusiones_OOF": errors})
    ranking = pd.DataFrame(rows).sort_values(
        ["confusiones_OOF", "clase_1", "clase_2"], ascending=[False, True, True]
    )
    return [ranking.iloc[0]["clase_1"], ranking.iloc[0]["clase_2"]], cm, ranking


def _plot_eda(df, raw, figdir):
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    clean_counts = df["Class"].value_counts().reindex(CLASSES)
    raw_counts = raw["Class"].value_counts().reindex(CLASSES)
    counts = pd.DataFrame({"Original": raw_counts, "Analítico": clean_counts}).reset_index()
    counts = counts.rename(columns={"Class": "Clase"}).melt("Clase", var_name="Conjunto", value_name="Cantidad")
    plt.figure(figsize=(10, 4.5))
    ax = sns.barplot(data=counts, x="Clase", y="Cantidad", hue="Conjunto")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    plt.title("Distribución de clases antes y después de quitar duplicados")
    plt.xticks(rotation=25)
    _save_fig(figdir / "01_distribucion_clases.png")

    X = df.drop(columns="Class")
    plt.figure(figsize=(11, 9))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlación entre las 16 características")
    _save_fig(figdir / "02_correlaciones.png")

    scaled = StandardScaler().fit_transform(X)
    projection = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), min(5000, len(X)), replace=False)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=projection[sample, 0],
        y=projection[sample, 1],
        hue=df["Class"].iloc[sample],
        hue_order=CLASSES,
        s=13,
        alpha=0.55,
    )
    plt.title("PCA exploratoria (no se usa para entrenar)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    _save_fig(figdir / "03_pca_exploratoria.png")


def _run_smo_checks(X_train, y_train, binary_pair, best_binary, resultdir, figdir):
    mask = y_train.isin(binary_pair)
    X_pair = X_train.loc[mask]
    y_pair = y_train.loc[mask]
    X_small, _, y_small, _ = train_test_split(
        X_pair,
        y_pair,
        train_size=400,
        stratify=y_pair,
        random_state=SEED,
    )
    scaler = StandardScaler().fit(X_small)
    xs = scaler.transform(X_small)
    ys = np.where(y_small.to_numpy() == binary_pair[1], 1.0, -1.0)

    C = min(float(best_binary["svc__C"]), 8.0)
    gamma = float(best_binary["svc__gamma"])
    own = SMOBinary(C=C, kernel="rbf", gamma=gamma, tol=1e-3).fit(xs, ys)
    ref = SVC(C=C, kernel="rbf", gamma=gamma, tol=1e-6).fit(xs, ys)
    K = own._kernel(xs, xs)
    alpha_ref = np.zeros(len(xs))
    alpha_ref[ref.support_] = np.abs(ref.dual_coef_[0])
    z_ref = alpha_ref * ys
    dual_ref = float(alpha_ref.sum() - 0.5 * z_ref @ K @ z_ref)
    own_score = own.decision_function(xs)
    ref_score = ref.decision_function(xs)
    comparison = {
        "n_subset": len(xs),
        "C": C,
        "gamma": gamma,
        "iterations": own.n_iter_,
        "converged": own.converged_,
        "m": own.m_,
        "M": own.M_,
        "m_minus_M": own.kkt_gap_,
        "equality_residual": own.equality_residual_,
        "dual_own": own.dual_objective_,
        "dual_libsvm": dual_ref,
        "max_abs_score_difference": float(np.max(np.abs(own_score - ref_score))),
        "prediction_agreement": float(np.mean(own.predict(xs) == ref.predict(xs))),
        "termination_reason": own.termination_reason_,
        "pointwise_kkt_max_violation": own.pointwise_kkt_violation(),
        "alpha_min": float(own.alpha_.min()),
        "alpha_max_minus_C": float(own.alpha_.max() - C),
        "bias_own": own.b_,
        "bias_libsvm": float(ref.intercept_[0]),
        "support_vectors_own": len(own.support_),
        "support_vectors_libsvm": len(ref.support_),
    }
    pd.DataFrame([comparison]).to_csv(resultdir / "e1_smo_vs_libsvm.csv", index=False)

    pd.DataFrame(
        {
            "iteration": np.arange(len(own.gap_history_)),
            "global_kkt_residual": own.gap_history_,
            "dual_objective": own.dual_history_,
        }
    ).to_csv(resultdir / "e1_convergencia_global.csv", index=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    history = np.maximum(own.gap_history_, 1e-12)
    ax1.semilogy(np.arange(len(history)), history, color="#176B87")
    ax1.axhline(own.tol, color="#B23A48", linestyle="--", label=f"tolerancia = {own.tol:g}")
    ax1.set_title("Residuo KKT global m − M")
    ax1.set_xlabel("iteración")
    ax1.set_ylabel("m − M (escala log)")
    ax1.legend()
    ax2.plot(np.arange(len(own.dual_history_)), own.dual_history_, color="#176B87", label="SMO propio")
    ax2.axhline(dual_ref, color="#B23A48", linestyle="--", label="LIBSVM")
    ax2.set_title("Objetivo dual D(α)")
    ax2.set_xlabel("iteración")
    ax2.legend()
    _save_fig(figdir / "06_convergencia_smo.png")
    return comparison


def _run_custom_multiclass_demo(X_train, y_train, X_test, y_test, C, gamma, resultdir):
    # Demostración acotada: la implementación NumPy usa matriz kernel O(n²).
    X_demo, _, y_demo, _ = train_test_split(
        X_train,
        y_train,
        train_size=420,
        stratify=y_train,
        random_state=SEED,
    )
    X_eval, _, y_eval, _ = train_test_split(
        X_test,
        y_test,
        train_size=210,
        stratify=y_test,
        random_state=SEED,
    )
    scaler = StandardScaler().fit(X_demo)
    xs = scaler.transform(X_demo)
    xe = scaler.transform(X_eval)
    params = {"C": min(float(C), 8.0), "kernel": "rbf", "gamma": float(gamma), "tol": 2e-3}
    rows = []
    for name, cls in [("OvR propio", SMOOneVsRest), ("OvO propio", SMOOneVsOne)]:
        model = cls(**params)
        start = perf_counter()
        model.fit(xs, y_demo.to_numpy())
        fit_s = perf_counter() - start
        start = perf_counter()
        pred = model.predict(xe)
        pred_s = perf_counter() - start
        rows.append(
            {
                "estrategia": name,
                "n_entrenamiento": len(xs),
                "n_prueba": len(xe),
                "accuracy": accuracy_score(y_eval, pred),
                "f1_macro": f1_score(y_eval, pred, average="macro"),
                "fit_s": fit_s,
                "predict_s": pred_s,
                "n_modelos_binarios": len(model.models_),
                "modelos_convergidos": int(
                    sum(
                        (m if isinstance(m, SMOBinary) else m[2]).converged_
                        for m in model.models_
                    )
                ),
                "max_kkt_residual": float(
                    max((m if isinstance(m, SMOBinary) else m[2]).kkt_gap_ for m in model.models_)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(resultdir / "e3_multiclase_smo_propio_subconjunto.csv", index=False)
    return result


def _run_timings(X_train, y_train, X_test, y_test, C, gamma, resultdir, figdir):
    repeats = 3
    sizes = [500, 1000, 2000, 4000, 8000]
    rows = []
    X_pred = X_test.iloc[:500]
    y_pred = y_test.iloc[:500]
    for n in sizes:
        Xi, _, yi, _ = train_test_split(
            X_train,
            y_train,
            train_size=n,
            stratify=y_train,
            random_state=SEED,
        )
        for strategy in ["OvR", "OvO"]:
            for repeat in range(repeats):
                metrics, _ = _fit_and_measure(
                    _explicit_model(strategy, C, gamma), Xi, yi, X_pred, y_pred
                )
                rows.append(
                    {
                        "n_train": n,
                        "strategy": strategy,
                        "repeat": repeat + 1,
                        "fit_s": metrics["fit_s"],
                        "predict_500_s": metrics["predict_s"],
                    }
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(resultdir / "e5_tiempos_repeticiones.csv", index=False)
    summary = (
        raw.groupby(["n_train", "strategy"])
        .agg(
            fit_median_s=("fit_s", "median"),
            fit_q1_s=("fit_s", lambda s: s.quantile(0.25)),
            fit_q3_s=("fit_s", lambda s: s.quantile(0.75)),
            predict_median_500_s=("predict_500_s", "median"),
            predict_q1_500_s=("predict_500_s", lambda s: s.quantile(0.25)),
            predict_q3_500_s=("predict_500_s", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    summary.to_csv(resultdir / "e5_tiempos_resumen.csv", index=False)

    fits = []
    for strategy, part in summary.groupby("strategy"):
        log_n = np.log(part["n_train"].to_numpy())
        log_t = np.log(part["fit_median_s"].to_numpy())
        theta, intercept = np.polyfit(log_n, log_t, 1)
        fitted = intercept + theta * log_n
        fits.append(
            {
                "strategy": strategy,
                "c": float(np.exp(intercept)),
                "theta": float(theta),
                "r2_log_log": float(r2_score(log_t, fitted)),
                "n_min": int(part["n_train"].min()),
                "n_max": int(part["n_train"].max()),
            }
        )
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(resultdir / "e5_ajuste_potencia.csv", index=False)

    plt.figure(figsize=(7.5, 4.5))
    for strategy, part in summary.groupby("strategy"):
        yerr = np.vstack(
            [
                part["fit_median_s"] - part["fit_q1_s"],
                part["fit_q3_s"] - part["fit_median_s"],
            ]
        )
        plt.errorbar(part["n_train"], part["fit_median_s"], yerr=yerr, marker="o", capsize=3, label=strategy)
    plt.title("E5: tiempo de entrenamiento vs. tamaño muestral")
    plt.xlabel("muestras de entrenamiento")
    plt.ylabel("segundos, mediana ± rango intercuartílico")
    plt.legend()
    _save_fig(figdir / "10_tiempo_vs_n.png")

    ordered = list(y_train.value_counts().index)
    k_rows = []
    for k in range(2, 8):
        subset = ordered[:k]
        mask_train = y_train.isin(subset)
        mask_test = y_test.isin(subset)
        Xi, _, yi, _ = train_test_split(
            X_train.loc[mask_train],
            y_train.loc[mask_train],
            train_size=700,
            stratify=y_train.loc[mask_train],
            random_state=SEED,
        )
        Xp = X_test.loc[mask_test].iloc[:300]
        yp = y_test.loc[mask_test].iloc[:300]
        for strategy in ["OvR", "OvO"]:
            for repeat in range(repeats):
                metrics, _ = _fit_and_measure(
                    _explicit_model(strategy, C, gamma), Xi, yi, Xp, yp
                )
                k_rows.append(
                    {
                        "k": k,
                        "classes": ",".join(subset),
                        "n_train": len(Xi),
                        "strategy": strategy,
                        "repeat": repeat + 1,
                        "fit_s": metrics["fit_s"],
                        "predict_300_s": metrics["predict_s"],
                    }
                )
    k_raw = pd.DataFrame(k_rows)
    k_raw.to_csv(resultdir / "e5_tiempos_por_k_repeticiones.csv", index=False)
    k_summary = (
        k_raw.groupby(["k", "strategy"])
        .agg(
            fit_median_s=("fit_s", "median"),
            fit_q1_s=("fit_s", lambda s: s.quantile(0.25)),
            fit_q3_s=("fit_s", lambda s: s.quantile(0.75)),
            predict_median_300_s=("predict_300_s", "median"),
            predict_q1_300_s=("predict_300_s", lambda s: s.quantile(0.25)),
            predict_q3_300_s=("predict_300_s", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    k_summary.to_csv(resultdir / "e5_tiempos_por_k_resumen.csv", index=False)
    plt.figure(figsize=(7.5, 4.5))
    sns.lineplot(data=k_summary, x="k", y="fit_median_s", hue="strategy", marker="o")
    plt.title("E5: costo al variar el número de clases (n=700 fijo)")
    plt.xlabel("número de clases k")
    plt.ylabel("segundos, mediana de 3 repeticiones")
    _save_fig(figdir / "11_tiempo_vs_k.png")
    return summary, fit_table, k_summary


def run_all(root):
    root = Path(root).resolve()
    figdir = root / "figures"
    resultdir = root / "results"
    figdir.mkdir(exist_ok=True)
    resultdir.mkdir(exist_ok=True)

    t_start = perf_counter()
    env = _environment(root)
    (resultdir / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[1/8] Cargando y limpiando datos...", flush=True)
    raw = pd.read_excel(root / "data" / "Dry_Bean_Dataset.xlsx")
    if raw.shape != (13611, 17) or "Class" not in raw.columns:
        raise ValueError(f"Dataset inesperado: {raw.shape}")
    missing = int(raw.isna().sum().sum())
    duplicates = int(raw.duplicated().sum())
    if missing != 0:
        raise ValueError("El archivo contiene valores faltantes y requiere revisar el preprocesamiento")
    df = raw.drop_duplicates().reset_index(drop=True)
    if duplicates != 68 or len(df) != 13543:
        raise ValueError("El conteo de duplicados no coincide con el dataset esperado")
    _plot_eda(df, raw, figdir)
    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    print("[2/8] CV multiclase de 5 pliegues con rejilla base 2...", flush=True)
    multi_cv_path = resultdir / "e4_cv_multiclase_rbf.csv"
    cache_meta = resultdir / "cache_meta.json"
    fingerprint = _fingerprint(root, {"stage": "multiclass_rbf_cv"})
    cache_ok = multi_cv_path.exists() and _cache_valid(cache_meta, fingerprint)
    if multi_cv_path.exists() and not cache_ok:
        print("      Caché descartada: datos, configuración o versiones no coinciden.", flush=True)
    if cache_ok:
        print("      Reutilizando resultados multiclase ya calculados.", flush=True)
        cv_multi = pd.read_csv(multi_cv_path)
        best_row = cv_multi.loc[cv_multi["mean_test_f1_macro"].idxmax()]
        best_C = float(best_row["param_svc__C"])
        best_gamma = float(best_row["param_svc__gamma"])
        best_cv_f1 = float(best_row["mean_test_f1_macro"])
        best_cv_accuracy = float(best_row["mean_test_accuracy"])
    else:
        multi_search, cv_multi = _staged_rbf_search(X_train, y_train)
        cv_multi.to_csv(multi_cv_path, index=False)
        cache_meta.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        best_C = float(multi_search.best_params_["svc__C"])
        best_gamma = float(multi_search.best_params_["svc__gamma"])
        best_cv_f1 = float(multi_search.best_score_)
        best_cv_accuracy = float(
            multi_search.cv_results_["mean_test_accuracy"][multi_search.best_index_]
        )
    best_multi_estimator = _pipeline("rbf", C=best_C, gamma=best_gamma)

    print("[3/8] Seleccionando la pareja binaria solo con predicciones OOF...", flush=True)
    oof_pred = cross_val_predict(
        best_multi_estimator, X_train, y_train, cv=_cv(), n_jobs=-1, method="predict"
    )
    binary_pair, oof_cm, pair_ranking = _pick_binary_pair(y_train, oof_pred)
    pair_ranking.to_csv(resultdir / "seleccion_pareja_binaria_oof.csv", index=False)
    pd.DataFrame(oof_cm, index=CLASSES, columns=CLASSES).to_csv(
        resultdir / "matriz_confusion_oof_entrenamiento.csv"
    )
    plt.figure(figsize=(8, 7))
    sns.heatmap(oof_cm, annot=True, fmt="d", cmap="Purples", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusión OOF en entrenamiento (selección de pareja binaria)")
    plt.xlabel("Predicción OOF")
    plt.ylabel("Clase real")
    _save_fig(figdir / "04_confusion_oof_entrenamiento.png")

    train_mask = y_train.isin(binary_pair)
    test_mask = y_test.isin(binary_pair)
    Xb_train, yb_train = X_train.loc[train_mask], y_train.loc[train_mask]
    Xb_test, yb_test = X_test.loc[test_mask], y_test.loc[test_mask]

    print(f"[4/8] E2 binario {binary_pair[0]} vs {binary_pair[1]}...", flush=True)
    binary_specs = {
        "Lineal": (_pipeline("linear"), {"svc__C": C_VALUES}),
        "Polinomial": (
            _pipeline("poly", coef0=1.0),
            {"svc__C": C_VALUES, "svc__degree": [2, 3], "svc__gamma": ["scale"]},
        ),
        "RBF": (_pipeline("rbf"), None),
    }
    binary_rows = []
    binary_searches = {}
    for name, (estimator, grid) in binary_specs.items():
        if name == "RBF":
            search, staged_results = _staged_rbf_search(Xb_train, yb_train)
        else:
            search = _grid_search(Xb_train, yb_train, estimator, grid)
            staged_results = pd.DataFrame(search.cv_results_).assign(search_phase="single")
        binary_searches[name] = search
        pred = search.predict(Xb_test)
        svc = search.best_estimator_.named_steps["svc"]
        row = {
            "kernel": name,
            "cv_accuracy": search.cv_results_["mean_test_accuracy"][search.best_index_],
            "cv_f1_macro": search.best_score_,
            "test_accuracy": accuracy_score(yb_test, pred),
            "test_f1_macro": f1_score(yb_test, pred, average="macro"),
            "support_vectors": int(svc.n_support_.sum()),
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        }
        binary_rows.append(row)
        staged_results.to_csv(resultdir / f"e2_cv_{name.lower()}.csv", index=False)
    binary_table = pd.DataFrame(binary_rows)
    binary_table.to_csv(resultdir / "e2_resultados_binarios.csv", index=False)
    plt.figure(figsize=(7.5, 4.5))
    melted = binary_table.melt(
        id_vars="kernel",
        value_vars=["cv_f1_macro", "test_f1_macro"],
        var_name="medición",
        value_name="F1 macro",
    )
    sns.barplot(data=melted, x="kernel", y="F1 macro", hue="medición")
    plt.ylim(0.75, 1.0)
    plt.title(f"E2: {binary_pair[0]} vs {binary_pair[1]}")
    _save_fig(figdir / "05_kernels_binarios.png")

    print("[5/8] E1: verificando SMO propio y extensiones OvR/OvO...", flush=True)
    smo = _run_smo_checks(
        X_train,
        y_train,
        binary_pair,
        binary_searches["RBF"].best_params_,
        resultdir,
        figdir,
    )
    custom_multi = _run_custom_multiclass_demo(
        X_train, y_train, X_test, y_test, best_C, best_gamma, resultdir
    )

    print("[6/8] E3: comparando OvR y OvO explícitos en el conjunto completo...", flush=True)
    multiclass = {}
    multiclass_predictions = {}
    for strategy in ["OvR", "OvO"]:
        model = _explicit_model(strategy, best_C, best_gamma)
        metrics, pred = _fit_and_measure(model, X_train, y_train, X_test, y_test)
        metrics["n_binary_models"] = len(model.named_steps["multi"].estimators_)
        metrics["support_vectors_sum"] = _support_count(model)
        multiclass[strategy] = metrics
        multiclass_predictions[strategy] = pred
        pd.DataFrame(
            classification_report(
                y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
            )
        ).T.to_csv(resultdir / f"e3_reporte_{strategy}.csv")
    pd.DataFrame(multiclass).T.to_csv(resultdir / "e3_comparacion_ovr_ovo.csv")
    bootstrap = _paired_bootstrap(
        y_test, multiclass_predictions["OvR"], multiclass_predictions["OvO"]
    )
    pd.DataFrame([bootstrap]).to_csv(resultdir / "e3_bootstrap_ovo_menos_ovr.csv", index=False)
    cm_ovr = confusion_matrix(y_test, multiclass_predictions["OvR"], labels=CLASSES)
    pd.DataFrame(cm_ovr, index=CLASSES, columns=CLASSES).to_csv(
        resultdir / "e3_matriz_confusion_ovr_test.csv"
    )
    cm_test = confusion_matrix(y_test, multiclass_predictions["OvO"], labels=CLASSES)
    pd.DataFrame(cm_test, index=CLASSES, columns=CLASSES).to_csv(
        resultdir / "e3_matriz_confusion_ovo_test.csv"
    )
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("E3: matriz de confusión OvO en test retenido")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    _save_fig(figdir / "07_confusion_ovo_test.png")

    print("[7/8] E4: sensibilidad de F1, exactitud y vectores soporte...", flush=True)
    cv_multi["C"] = cv_multi["param_svc__C"].astype(float)
    cv_multi["gamma"] = cv_multi["param_svc__gamma"].astype(float)
    support_path = resultdir / "e4_vectores_soporte.csv"
    if support_path.exists() and cache_ok:
        support_table = pd.read_csv(support_path)
    else:
        evaluated_pairs = cv_multi[["C", "gamma"]].drop_duplicates().itertuples(index=False, name=None)
        support_rows = []
        for C, gamma in evaluated_pairs:
            model = _pipeline("rbf", C=C, gamma=gamma).fit(X_train, y_train)
            support_rows.append(
                {"C": C, "gamma": gamma, "support_vectors": int(model.named_steps["svc"].support_.size)}
            )
        support_table = pd.DataFrame(support_rows)
        support_table.to_csv(support_path, index=False)
    for metric, filename, title in [
        ("mean_test_f1_macro", "08_heatmap_f1_cv.png", "E4: F1 macro medio en CV"),
        ("mean_test_accuracy", "09_heatmap_accuracy_cv.png", "E4: exactitud media en CV"),
    ]:
        plt.figure(figsize=(9, 6))
        table = cv_multi.pivot_table(index="gamma", columns="C", values=metric, aggfunc="max")
        sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(title)
        _save_fig(figdir / filename)
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        support_table.pivot(index="gamma", columns="C", values="support_vectors"),
        annot=True,
        fmt=".0f",
        cmap="YlOrBr",
    )
    plt.title("E4: vectores soporte al variar C y gamma")
    _save_fig(figdir / "09b_heatmap_vectores_soporte.png")

    print("[8/8] E5: tiempos repetidos y ajuste T(m)=c·m^theta...", flush=True)
    timing_summary, power_fit, k_summary = _run_timings(
        X_train, y_train, X_test, y_test, best_C, best_gamma, resultdir, figdir
    )

    raw_counts = raw["Class"].value_counts().reindex(CLASSES).to_dict()
    clean_counts = df["Class"].value_counts().reindex(CLASSES).to_dict()
    metrics = {
        "data": {
            "raw_rows": len(raw),
            "duplicates_removed": duplicates,
            "analysis_rows": len(df),
            "features": X.shape[1],
            "missing_values": missing,
            "raw_class_counts": raw_counts,
            "analysis_class_counts": clean_counts,
        },
        "split": {
            "train": len(X_train),
            "test": len(X_test),
            "test_fraction": 0.20,
            "stratified": True,
            "seed": SEED,
            "cv_folds": 5,
        },
        "search": {
            "method": "base-2 staged: coarse grid plus local refinement",
            "coarse_C_exponents": COARSE_C_EXP,
            "coarse_gamma_exponents": COARSE_GAMMA_EXP,
            "refinement": "exponente ganador y vecinos ±1",
        },
        "multiclass_cv": {
            "best_C": best_C,
            "best_gamma": best_gamma,
            "best_cv_f1_macro": best_cv_f1,
            "best_cv_accuracy": best_cv_accuracy,
        },
        "binary_selection": {
            "method": "mayor confusión simétrica en predicciones OOF de 5 pliegues del entrenamiento",
            "classes": binary_pair,
            "oof_confusions": int(pair_ranking.iloc[0]["confusiones_OOF"]),
            "train_rows": len(Xb_train),
            "test_rows": len(Xb_test),
        },
        "binary_results": binary_table.to_dict(orient="records"),
        "smo": smo,
        "custom_multiclass_demo": custom_multi.to_dict(orient="records"),
        "multiclass_test": multiclass,
        "timing_power_fit": power_fit.to_dict(orient="records"),
        "e3_bootstrap_ovo_minus_ovr": bootstrap,
        "environment": env,
        "total_runtime_s": perf_counter() - t_start,
        "notes": {
            "no_imputation": "No hay faltantes; se usa StandardScaler dentro del pipeline.",
            "test_policy": "El test se usa una sola vez para métricas finales.",
            "custom_scope": "SMO y wrappers propios se verifican en subconjuntos por su costo O(n²); los resultados completos usan SVC/LIBSVM.",
            "timing": "Tres repeticiones; se reporta mediana e IQR. theta es empírico, no complejidad teórica.",
        },
    }
    (resultdir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Listo. Resultados guardados en figures/ y results/.", flush=True)
    return metrics
