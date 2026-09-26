# Resultados y archivos generados

Esta referencia relaciona cada etapa experimental con sus tablas y figuras. Los valores se generan al ejecutar `run_experiments.py` y también se muestran en `OPTI.ipynb`.

## Datos y metodología

- `results/metrics.json`: 13 611 filas originales, 68 duplicados, 13 543 filas analíticas, 16 variables y 0 faltantes.
- `figures/01_distribucion_clases.png`: distribución antes y después de retirar duplicados.
- `figures/02_correlacion.png`: correlación entre las variables numéricas.
- `figures/03_pca_exploratoria.png`: proyección PCA usada únicamente para exploración visual.
- Partición estratificada 80/20 con semilla 42 y validación cruzada estratificada de cinco pliegues.
- Estandarización dentro del pipeline para evitar fuga de información.

## E1 — Implementación de SMO

- `results/e1_smo_vs_libsvm.csv`: objetivo dual, brecha KKT, residuo de igualdad, diferencia máxima de puntuaciones, concordancia y vectores soporte.
- `figures/06_convergencia_smo.png`: evolución de la brecha KKT.
- La comprobación se realiza sobre 400 filas de entrenamiento debido al costo cuadrático de la matriz kernel.

## E2 — Clasificación binaria

- `results/seleccion_pareja_binaria_oof.csv`: ranking de confusiones por pareja.
- `figures/04_confusion_oof_entrenamiento.png`: matriz OOF usada para seleccionar DERMASON–SIRA sin consultar el test.
- `results/e2_resultados_binarios.csv`: comparación de los kernels lineal, polinomial y RBF.
- `figures/05_kernels_binarios.png`: F1 macro de validación y prueba.

RBF obtuvo el mayor F1 macro, aunque la diferencia frente al kernel polinomial fue pequeña.

## E3 — Clasificación multiclase

- `results/e3_multiclase_smo_propio_subconjunto.csv`: comprobación de OvR y OvO con la implementación NumPy.
- `results/e3_comparacion_ovr_ovo.csv`: comparación completa sobre el conjunto de prueba.
- `results/e3_reporte_OvR.csv` y `results/e3_reporte_OvO.csv`: métricas por clase.
- `results/e3_matriz_confusion_ovo_test.csv` y `figures/07_confusion_ovo_test.png`: matriz de confusión de OvO.

OvO obtuvo un F1 macro ligeramente mayor en esta partición. La diferencia no permite afirmar que una estrategia sea siempre superior.

## E4 — Sensibilidad de hiperparámetros

- `results/e4_cv_multiclase_rbf.csv`: resultados de validación cruzada.
- `results/e4_vectores_soporte.csv`: cantidad de vectores soporte por combinación.
- `figures/08_heatmap_f1_cv.png`, `figures/09_heatmap_accuracy_cv.png` y `figures/09b_heatmap_vectores_soporte.png`: mapas de calor.

El mejor punto evaluado fue `C=64` y `gamma=0.015625`, con F1 macro medio de 0.9448 en validación cruzada.

## E5 — Costo computacional

- `results/e5_tiempos_repeticiones.csv` y `results/e5_tiempos_por_k_repeticiones.csv`: mediciones individuales.
- `results/e5_tiempos_resumen.csv` y `results/e5_tiempos_por_k_resumen.csv`: medianas y rangos intercuartílicos.
- `results/e5_ajuste_potencia.csv`: ajuste empírico `T(m)=c·m^theta`.
- `figures/10_tiempo_vs_n.png` y `figures/11_tiempo_vs_k.png`: evolución del tiempo respecto al tamaño y al número de clases.

Los tiempos dependen del equipo y de la carga del sistema. El exponente `theta` describe únicamente el intervalo medido y no representa una demostración de complejidad asintótica.

## Alcance

- La implementación NumPy se valida en subconjuntos; la evaluación completa utiliza `SVC`/LIBSVM.
- El conjunto de prueba corresponde a una sola partición retenida.
- Las entradas son características geométricas extraídas previamente de imágenes, no imágenes originales.
