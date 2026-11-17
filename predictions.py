
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SEED = 42  # misma semilla que src/experiments.py
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
STRATEGIES = ["OvR", "OvO"]


def load_test_split():
    # Mismo preprocesamiento y partición que run_all: sin duplicados, 80/20 estratificado.
    df = pd.read_excel(ROOT / "data" / "Dry_Bean_Dataset.xlsx")
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns="Class"), df["Class"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    return X_test, y_test


def main():
    X_test, y_test = load_test_split()

    cms, summary = {}, []
    for strategy in STRATEGIES:
        model = joblib.load(ROOT / "results" / f"modelo_svm_{strategy}.joblib")
        pred = model.predict(X_test)
        cms[strategy] = confusion_matrix(y_test, pred, labels=CLASSES)
        summary.append(
            {
                "estrategia": strategy,
                "accuracy": accuracy_score(y_test, pred),
                "f1_macro": f1_score(y_test, pred, average="macro"),
                "errores": int((pred != y_test).sum()),
            }
        )

    print(f"Test retenido: {len(y_test)} muestras\n")
    print(pd.DataFrame(summary).to_string(index=False, float_format="%.4f"))

    diff = cms["OvO"] - cms["OvR"]
    for strategy in STRATEGIES:
        print(f"\nMatriz de confusión {strategy} (filas = real, columnas = predicción):")
        print(pd.DataFrame(cms[strategy], index=CLASSES, columns=CLASSES))
    print("\nDiferencia OvO - OvR (diagonal > 0 = OvO acierta más; fuera de diagonal < 0 = OvO confunde menos):")
    print(pd.DataFrame(diff, index=CLASSES, columns=CLASSES))

    recall = {s: np.diag(cms[s]) / cms[s].sum(axis=1) for s in STRATEGIES}
    print("\nRecall por clase:")
    print(
        pd.DataFrame(recall, index=CLASSES)
        .assign(diferencia=lambda d: d["OvO"] - d["OvR"])
        .to_string(float_format="%.4f")
    )

    # Figura: OvR | OvO | diferencia
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    vmax = max(cm.max() for cm in cms.values())
    for ax, strategy, row in zip(axes, STRATEGIES, summary):
        sns.heatmap(
            cms[strategy], annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False,
        )
        ax.set_title(f"{strategy}  (acc={row['accuracy']:.4f}, F1={row['f1_macro']:.4f})")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
    # Color = mejora de OvO: en la diagonal más aciertos es mejor; fuera de ella,
    # menos confusiones es mejor. Se invierte el signo fuera de la diagonal solo
    # para colorear; los números anotados son la diferencia real OvO - OvR.
    improvement = np.where(np.eye(len(CLASSES), dtype=bool), diff, -diff)
    lim = max(1, int(np.abs(diff).max()))
    sns.heatmap(
        improvement, annot=diff, fmt="d", cmap="RdBu", center=0, vmin=-lim, vmax=lim,
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2],
        cbar_kws={"label": "← mejor OvR      mejor OvO →"},
    )
    axes[2].set_title("Diferencia OvO − OvR (azul = OvO mejor, rojo = OvR mejor)")
    axes[2].set_xlabel("Predicción")
    axes[2].set_ylabel("Clase real")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    out = ROOT / "figures" / "12_confusion_ovr_vs_ovo.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"\nFigura guardada en {out.relative_to(ROOT)}")
    plt.show()


if __name__ == "__main__":
    main()
