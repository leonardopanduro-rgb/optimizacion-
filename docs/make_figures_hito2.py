"""Genera las figuras del informe del Hito II a partir de docs/resultados_recibidos/.

Uso: python docs/make_figures_hito2.py
Salida: docs/figuras_hito2/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "resultados_recibidos"
OUT = ROOT / "docs" / "figuras_hito2"
OUT.mkdir(exist_ok=True)

AZUL, NARANJA, ROJO = "#1F6FB2", "#C8702A", "#B23A48"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()


# Fig. 06: residuo KKT global y objetivo dual
conv = pd.read_csv(SRC / "e1_convergencia_global.csv")
e1 = pd.read_csv(SRC / "e1_smo_vs_libsvm.csv").iloc[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogy(conv["iteration"], np.maximum(conv["global_kkt_residual"], 1e-12), color=AZUL, lw=1.5)
ax1.axhline(1e-3, color=ROJO, ls="--", label="tolerancia = 0.001")
ax1.set(title="Residuo KKT global m − M", xlabel="iteración", ylabel="m − M (escala log)")
ax1.legend(frameon=False)
ax2.plot(conv["iteration"], conv["dual_objective"], color=AZUL, lw=1.5, label="SMO propio")
ax2.axhline(e1["dual_libsvm"], color=ROJO, ls="--", label="LIBSVM")
ax2.set(title="Objetivo dual D(α)", xlabel="iteración")
ax2.legend(frameon=False, loc="lower right")
save("06_convergencia_smo.png")

# E2: F1 de CV y de prueba por kernel
e2 = pd.read_csv(SRC / "e2_resultados_binarios.csv")
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(e2))
ax.scatter(e2["cv_f1_macro"], y + 0.12, color=AZUL, s=60, label="Validación cruzada", zorder=3)
ax.scatter(e2["test_f1_macro"], y - 0.12, color=NARANJA, s=60, label="Prueba", zorder=3)
for yi, (cv, te) in enumerate(zip(e2["cv_f1_macro"], e2["test_f1_macro"])):
    ax.annotate(f"{cv:.4f}", (cv, yi + 0.12), xytext=(6, -3), textcoords="offset points")
    ax.annotate(f"{te:.4f}", (te, yi - 0.12), xytext=(6, -3), textcoords="offset points")
ax.set_yticks(y, e2["kernel"])
ax.invert_yaxis()
ax.set_xlim(0.905, 0.94)
ax.set_xlabel("F1 macro (escala ampliada para comparar valores)")
ax.set_title("E2: DERMASON − SIRA")
ax.grid(axis="x", alpha=0.3)
ax.legend(frameon=False, loc="lower left")
save("e2_kernels_puntos.png")

# E4: mapa de calor de F1 macro CV
cv = pd.read_csv(SRC / "e4_cv_multiclase_rbf.csv")
tabla = cv.pivot_table(index="param_svc__gamma", columns="param_svc__C", values="mean_test_f1_macro", aggfunc="max")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
im = ax.imshow(tabla.to_numpy(), cmap="YlGnBu", aspect="auto", origin="upper")
ax.set_xticks(range(tabla.shape[1]), [f"$2^{{{int(np.log2(c))}}}$" for c in tabla.columns])
ax.set_yticks(range(tabla.shape[0]), [f"$2^{{{int(np.log2(g))}}}$" for g in tabla.index])
ax.set(xlabel="Penalización C", ylabel="Parámetro gamma", title="E4: F1 macro medio en validación cruzada")
best = np.nanargmax(tabla.to_numpy())
bi, bj = np.unravel_index(best, tabla.shape)
for i in range(tabla.shape[0]):
    for j in range(tabla.shape[1]):
        v = tabla.iat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=ROJO, lw=2))
fig.colorbar(im, ax=ax, label="Promedio en validación cruzada")
fig.text(0.5, -0.02, "Blanco: combinación no evaluada. Borde rojo: mayor promedio observado.", ha="center", fontsize=8)
save("e4_heatmap_f1.png")

# E3: diferencia OvO − OvR por semilla con IC bootstrap
s = pd.read_csv(SRC / "hito2_remuestreo_semillas.csv")
fig, ax = plt.subplots(figsize=(8.5, 5))
d, lo, hi = (s[c] * 100 for c in ["e3_diff_ovo_minus_ovr", "e3_boot_ci_low", "e3_boot_ci_high"])
ax.errorbar(d, s["seed"], xerr=[d - lo, hi - d], fmt="o", color=AZUL, capsize=3, lw=1.2, ms=6)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_yticks(s["seed"])
ax.invert_yaxis()
ax.set(xlabel="Diferencia de F1 macro: OvO − OvR (puntos porcentuales)", ylabel="Semilla de la partición",
       title="E3: la ventaja cambia de signo entre particiones")
fig.text(0.5, -0.03, "Barras: IC bootstrap pareado del 95 %, condicionado a cada par de modelos.\n"
         "Las particiones reutilizan datos; estos diez intervalos no son independientes.", ha="center", fontsize=8)
save("e3_forest_semillas.png")

# E5: tiempos con mediana y cuartiles
t = pd.read_csv(SRC / "e5_tiempos_resumen.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for strat, color in [("OvR", AZUL), ("OvO", NARANJA)]:
    p = t[t.strategy == strat]
    ax1.errorbar(p.n_train, p.fit_median_s, yerr=[p.fit_median_s - p.fit_q1_s, p.fit_q3_s - p.fit_median_s],
                 marker="o", color=color, capsize=3, label=strat)
    ax2.errorbar(p.n_train, p.predict_median_500_s,
                 yerr=[p.predict_median_500_s - p.predict_q1_500_s, p.predict_q3_500_s - p.predict_median_500_s],
                 marker="o", color=color, capsize=3, label=strat)
ax1.set(title="Entrenamiento", xlabel="Filas de entrenamiento (7 clases)", ylabel="Tiempo (segundos)")
ax2.set(title="Predicción de 500 filas", xlabel="Filas de entrenamiento (7 clases)", ylabel="Tiempo (segundos)")
for ax in (ax1, ax2):
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
fig.suptitle("E5: mediana y cuartiles Q1−Q3 de tres repeticiones")
save("e5_tiempos.png")

# Matriz de confusión OvO en prueba
cm = pd.read_csv(SRC / "e3_matriz_confusion_ovo_test.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm.to_numpy(), cmap="Blues")
ax.set_xticks(range(len(cm)), cm.columns, rotation=35, ha="right")
ax.set_yticks(range(len(cm)), cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        v = cm.iat[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if v > cm.to_numpy().max() / 2 else "black")
ax.set(xlabel="Predicción", ylabel="Clase real", title="E3: matriz de confusión OvO en el test")
save("e3_confusion_ovo_test.png")
print("Figuras guardadas en", OUT)
