# Guía para actualizar después el informe LaTeX

El PDF actual funcionó como especificación metodológica. Esta guía indica qué resultados del repositorio deben reemplazar o completar las secciones de resultados. No se modificó todavía el documento LaTeX.

## Dataset y metodología

- Usar `results/metrics.json` para declarar 13 611 filas originales, 68 duplicados, 13 543 filas analíticas, 16 variables y 0 faltantes.
- Actualizar la clase HOROZ de 1 928 a 1 860 solo cuando se hable del conjunto limpio; conservar los conteos originales si se describe la fuente.
- Aclarar que no hay imputación. El pipeline contiene `StandardScaler` y `SVC`.
- Indicar split estratificado 80/20, semilla 42 y CV estratificada de 5 pliegues.
- Describir la búsqueda base 2 escalonada: malla gruesa más refinamiento de exponentes vecinos.

## E1 — Implementación propia

- Tabla: `results/e1_smo_vs_libsvm.csv`.
- Figura: `figures/06_convergencia_smo.png`.
- Reportar objetivo dual, `m`, `M`, `m-M`, residuo `|y^T alpha|`, diferencia máxima de scores, concordancia y vectores soporte.
- Escribir que SMO propio usa máxima violación y que la comparación se realiza en 400 filas de entrenamiento.

## Selección del caso binario y E2

- Tabla de ranking: `results/seleccion_pareja_binaria_oof.csv`.
- Figura justificativa: `figures/04_confusion_oof_entrenamiento.png`.
- La pareja correcta es DERMASON–SIRA, elegida con predicciones OOF del entrenamiento.
- Tabla de kernels: `results/e2_resultados_binarios.csv`.
- Figura: `figures/05_kernels_binarios.png`.
- No afirmar que una diferencia pequeña prueba superioridad universal.

## E3 — OvR y OvO

- Demo propia: `results/e3_multiclase_smo_propio_subconjunto.csv`.
- Comparación completa: `results/e3_comparacion_ovr_ovo.csv`.
- Métricas por clase: `results/e3_reporte_OvR.csv` y `results/e3_reporte_OvO.csv`.
- Matriz/figura: `results/e3_matriz_confusion_ovo_test.csv` y `figures/07_confusion_ovo_test.png`.
- Redacción prudente: OvO fue ligeramente mejor en esta partición; la diferencia de F1 macro fue cercana a 0.0012.

## E4 — Sensibilidad

- Resultados CV: `results/e4_cv_multiclase_rbf.csv`.
- Vectores soporte: `results/e4_vectores_soporte.csv`.
- Figuras: `figures/08_heatmap_f1_cv.png`, `figures/09_heatmap_accuracy_cv.png` y `figures/09b_heatmap_vectores_soporte.png`.
- El mejor punto fue `C=64` y `gamma=0.015625`, con F1 macro CV `0.9448`.

## E5 — Costos

- Datos crudos: `results/e5_tiempos_repeticiones.csv` y `results/e5_tiempos_por_k_repeticiones.csv`.
- Resúmenes: `results/e5_tiempos_resumen.csv` y `results/e5_tiempos_por_k_resumen.csv`.
- Ajuste de potencia: `results/e5_ajuste_potencia.csv`.
- Figuras: `figures/10_tiempo_vs_n.png` y `figures/11_tiempo_vs_k.png`.
- Reportar tres repeticiones, mediana e IQR. `theta` es empírico, no complejidad teórica.

## Limitaciones que deben quedar visibles

- SMO propio y wrappers propios se verifican en subconjuntos; la evaluación completa usa LIBSVM.
- Se usa una sola partición de test retenida.
- Los tiempos cambian con hardware y carga del sistema.
- Las entradas son características ya extraídas, no imágenes crudas.
